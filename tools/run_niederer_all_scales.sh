#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NP="${NP:-1}"
BENCHMARK_PROBES="${BENCHMARK_PROBES:-1}"

run_scale() {
  local label="$1"
  local nx="$2"
  local ny="$3"
  local nz="$4"
  local out_dir="$5"
  local cfg="$6"

  echo "==> [${label}] generating mesh/fibers (${nx}x${ny}x${nz}) into ${out_dir}"
  mpirun -np 1 ./build/generate_niederer_case "${out_dir}" --nx "${nx}" --ny "${ny}" --nz "${nz}"

  echo "==> [${label}] running monodomain with ${cfg} (NP=${NP}, probes=${BENCHMARK_PROBES})"
  mpirun -np "${NP}" ./build/monodomain --config "${cfg}" --benchmark-probes "${BENCHMARK_PROBES}"
}

run_scale "h=0.5mm" 40 14 6 "benchmarks/niederer_h050" "config/niederer_50ms_h050.options"
run_scale "h=0.2mm" 100 35 15 "benchmarks/niederer_h020" "config/niederer_50ms_h020.options"
run_scale "h=0.1mm" 200 70 30 "benchmarks/niederer_h010" "config/niederer_50ms_h010.options"

echo "==> all Niederer scales finished."
