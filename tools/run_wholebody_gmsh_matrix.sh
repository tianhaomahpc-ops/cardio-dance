#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCALES_MM=(${SCALES_MM:-0.5 0.2 0.1})
RANKS=(${RANKS:-1 2 4 8})
SOLVERS=(${SOLVERS:-boomer asm})
PADDING_MM="${PADDING_MM:-1.0}"
T_END_MS="${T_END_MS:-100}"
DT_PDE_MS="${DT_PDE_MS:-0.02}"
DT_ODE_MS="${DT_ODE_MS:-0.01}"
KSP_MAX_IT="${KSP_MAX_IT:-2000}"
KSP_RTOL="${KSP_RTOL:-1e-8}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-10000}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-0}"
PETSC_ASM_OPTIONS="${PETSC_ASM_OPTIONS:--mono_ksp_type cg -mono_pc_type asm -mono_pc_asm_overlap 1}"
WHOLEBODY_SOLVE_EVERY_STEP="${WHOLEBODY_SOLVE_EVERY_STEP:-0}"
PRUNE_FIELD_OUTPUT="${PRUNE_FIELD_OUTPUT:-0}"
MESH_OPTIMIZE="${MESH_OPTIMIZE:-0}"

MESH_DIR="${MESH_DIR:-benchmarks/wholebody_gmsh}"
RUN_ROOT="${RUN_ROOT:-output/wholebody_gmsh_matrix}"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
CONSISTENCY_CSV="${RUN_ROOT}/consistency.csv"
REGENERATE_MESH="${REGENERATE_MESH:-0}"

mkdir -p "$MESH_DIR" "$RUN_ROOT"
printf "scale_mm,solver,np,status,elapsed_sec,mesh_path,out_dir,log_path,finished_ms,vm_min,vm_max,vm_mean,vm_l2,iion_min,iion_max,iion_mean,iion_l2,ue_min,ue_max,ue_mean,ue_l2,ut_min,ut_max,ut_mean,ut_l2,heart_cycles,torso_cycles\n" > "$SUMMARY_CSV"
printf "scale_mm,solver,np,vm_mean_diff_vs_np1,iion_mean_diff_vs_np1,ue_mean_diff_vs_np1,ut_mean_diff_vs_np1\n" > "$CONSISTENCY_CSV"

scale_tag() {
  case "$1" in
    0.5) echo "h050" ;;
    0.2) echo "h020" ;;
    0.1) echo "h010" ;;
    *)
      local scaled
      scaled="$(awk -v v="$1" 'BEGIN{printf "%03d", int(v * 100 + 0.5)}')"
      echo "h${scaled}"
      ;;
  esac
}

extract_finished_ms() {
  local log_path="$1"
  local line
  line="$(grep -F "[monodomain] finished at t =" "$log_path" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "NA"
  else
    echo "$line" | sed -E 's/.*t = ([0-9eE+.-]+) ms.*/\1/'
  fi
}

extract_stats_values() {
  local log_path="$1"
  local key="$2"
  local line
  line="$(grep -F "[monodomain] ${key} stats:" "$log_path" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "NA,NA,NA,NA"
  else
    echo "$line" | sed -E 's/.*min=([^,]+), max=([^,]+), mean=([^,]+), l2=([^,]+).*/\1,\2,\3,\4/'
  fi
}

for h in "${SCALES_MM[@]}"; do
  tag="$(scale_tag "$h")"
  mesh_path="${MESH_DIR}/wholebody_${tag}.msh"
  if [[ "$REGENERATE_MESH" == "1" || ! -f "$mesh_path" ]]; then
    echo "==> [${tag}] generate Gmsh mesh (h=${h}mm, padding=${PADDING_MM}mm)"
    python3 tools/generate_conforming_wholebody_gmsh.py \
      --out-mesh "$mesh_path" \
      --h-mm "$h" \
      --padding-mm "$PADDING_MM" \
      --heart-attr 1 \
      --torso-attr 2 \
      --optimize "$MESH_OPTIMIZE"
  else
    echo "==> [${tag}] reuse mesh ${mesh_path}"
  fi
  mesh_abs="$(cd "$(dirname "$mesh_path")" && pwd)/$(basename "$mesh_path")"

  for solver in "${SOLVERS[@]}"; do
    use_petsc=0
    use_boomer=0
    case "$solver" in
      boomer)
        use_petsc=0
        use_boomer=1
        ;;
      asm)
        use_petsc=1
        use_boomer=0
        ;;
      *)
        echo "!! unknown solver '${solver}', expected boomer or asm"
        continue
        ;;
    esac

    for np in "${RANKS[@]}"; do
      run_id="${tag}_${solver}_np${np}"
      out_dir="${RUN_ROOT}/${run_id}"
      ckpt_dir="checkpoint/wholebody_gmsh_matrix/${run_id}"
      cfg_path="${out_dir}/run.options"
      log_path="${out_dir}/run.log"
      mkdir -p "$out_dir" "$ckpt_dir"

      cat > "$cfg_path" <<EOF
enable_wholebody=1
use_conforming_wholebody=1
wholebody_mesh_path=${mesh_path}
heart_volume_attrs=1
torso_volume_attrs=2
use_fiber_gf=0
cm_uF_per_mm2=0.01
chi_per_mm=140.0
sigma_f_mS_per_mm=0.1334
sigma_s_mS_per_mm=0.0176
sigma_n_mS_per_mm=0.0176
sigma_i_f_mS_per_mm=0.174
sigma_i_s_mS_per_mm=0.019
sigma_i_n_mS_per_mm=0.019
sigma_e_f_mS_per_mm=0.625
sigma_e_s_mS_per_mm=0.236
sigma_e_n_mS_per_mm=0.236
sigma_torso_mS_per_mm=0.2
interface_map_max_dist_mm=1.0
torso_dirichlet_penalty=1e6
dt_pde_ms=${DT_PDE_MS}
dt_ode_ms=${DT_ODE_MS}
t_end_ms=${T_END_MS}
stim_start_ms=0.0
stim_end_ms=2.0
stim_amp=-15.0
stim_xmin_mm=0.0
stim_xmax_mm=1.5
stim_ymin_mm=0.0
stim_ymax_mm=1.5
stim_zmin_mm=0.0
stim_zmax_mm=1.5
use_petsc=${use_petsc}
use_hypre_boomeramg=${use_boomer}
wholebody_solve_every_step=${WHOLEBODY_SOLVE_EVERY_STEP}
ksp_max_it=${KSP_MAX_IT}
ksp_rtol=${KSP_RTOL}
output_stride=${OUTPUT_STRIDE}
checkpoint_stride=${CHECKPOINT_STRIDE}
output_dir=${out_dir}
checkpoint_dir=${ckpt_dir}
EOF

      echo "==> [${run_id}] run solver=${solver}, np=${np}, t_end=${T_END_MS}ms"
      start_ts="$(date +%s)"
      if [[ "$solver" == "asm" ]]; then
        if PETSC_OPTIONS="${PETSC_ASM_OPTIONS}" mpirun -np "$np" ./build/monodomain --config "$cfg_path" > "$log_path" 2>&1; then
          status="ok"
        else
          status="fail"
        fi
      else
        if mpirun -np "$np" ./build/monodomain --config "$cfg_path" > "$log_path" 2>&1; then
          status="ok"
        else
          status="fail"
        fi
      fi
      end_ts="$(date +%s)"
      elapsed="$((end_ts - start_ts))"

      finished_ms="$(extract_finished_ms "$log_path")"
      vm_stats="$(extract_stats_values "$log_path" "Vm")"
      iion_stats="$(extract_stats_values "$log_path" "Iion")"
      ue_stats="$(extract_stats_values "$log_path" "ue")"
      ut_stats="$(extract_stats_values "$log_path" "uT")"
      heart_cycles=0
      torso_cycles=0
      if [[ -d "${out_dir}/heart/monodomain" ]]; then
        heart_cycles="$(find "${out_dir}/heart/monodomain" -maxdepth 1 -type d -name 'Cycle*' | wc -l | tr -d ' ')"
      fi
      if [[ -d "${out_dir}/torso/torso_potential" ]]; then
        torso_cycles="$(find "${out_dir}/torso/torso_potential" -maxdepth 1 -type d -name 'Cycle*' | wc -l | tr -d ' ')"
      fi
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$h" "$solver" "$np" "$status" "$elapsed" "$mesh_abs" "$out_dir" "$log_path" "$finished_ms" \
        "$vm_stats" "$iion_stats" "$ue_stats" "$ut_stats" "$heart_cycles" "$torso_cycles" >> "$SUMMARY_CSV"

      if [[ "$PRUNE_FIELD_OUTPUT" == "1" ]]; then
        rm -rf "${out_dir}/heart" "${out_dir}/torso"
      fi
    done
  done
done

awk -F, '
NR==1 { next }
{
  key = $1 "," $2;
  scale = $1;
  solver = $2;
  np = $3;
  status = $4;
  if (status != "ok") next;
  vm_mean = $12;
  iion_mean = $16;
  ue_mean = $20;
  ut_mean = $24;
  if (np == "1") {
    vm_ref[key] = vm_mean + 0.0;
    iion_ref[key] = iion_mean + 0.0;
    ue_ref[key] = ue_mean + 0.0;
    ut_ref[key] = ut_mean + 0.0;
  } else if (key in vm_ref) {
    vm_err = vm_mean - vm_ref[key];
    iion_err = iion_mean - iion_ref[key];
    ue_err = ue_mean - ue_ref[key];
    ut_err = ut_mean - ut_ref[key];
    printf "%s,%s,%s,%.12g,%.12g,%.12g,%.12g\n", scale, solver, np, vm_err, iion_err, ue_err, ut_err;
  }
}
' "$SUMMARY_CSV" >> "$CONSISTENCY_CSV"

echo "==> done"
echo "summary: ${SUMMARY_CSV}"
echo "consistency: ${CONSISTENCY_CSV}"
