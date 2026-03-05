#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

NP=${NP:-1}
CONFIG=${CONFIG:-config/niederer_purkinje_avdelay.options}
MONODOMAIN_EXE=${MONODOMAIN_EXE:-./build/monodomain}
MESH_TOOL=${MESH_TOOL:-./build/split_niederer_x_regions_av}

echo "[validation] generate segmented mesh for AV-delay + low-conductivity regions"
mpirun -np 1 "$MESH_TOOL" \
  --in-mesh benchmarks/niederer/niederer_benchmark.mesh \
  --out-mesh benchmarks/niederer/niederer_purkinje_avdelay.mesh

echo "[validation] run monodomain with Purkinje+PVJ"
mpirun -np "$NP" "$MONODOMAIN_EXE" --config "$CONFIG" --benchmark-probes 1

OUT_DIR=$(awk -F= '/^output_dir[ 	]*=/{print $2}' "$CONFIG" | tail -n1 | xargs)
if [[ -z "$OUT_DIR" ]]; then
  echo "[validation] cannot infer output_dir from $CONFIG" >&2
  exit 2
fi

PROBE_CSV="$OUT_DIR/probe_vm.csv"
if [[ ! -f "$PROBE_CSV" ]]; then
  echo "[validation] missing probe csv: $PROBE_CSV" >&2
  exit 3
fi

echo "[validation] extract activation times"
python3 tools/extract_activation_times.py \
  --csv "$PROBE_CSV" \
  --probes "Atria,AVDelay,Fibrosis,Ventricle" \
  --out "$OUT_DIR/activation_times.json"

echo "[validation] done: $OUT_DIR"
