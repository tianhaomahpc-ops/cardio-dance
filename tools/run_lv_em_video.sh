#!/usr/bin/env bash
# End-to-end LV electromechanical demo + MP4 render.
#
# Steps:
#   1) generate the truncated prolate-ellipsoid LV mesh + helical fibers
#   2) run monodomain + Land2017 + Holzapfel-Ogden for `t_end_ms`
#   3) render ParaView frames as a side-by-side Vm/Ta MP4
#
# Env knobs (with defaults):
#   NP=4              MPI processes
#   N_PHI=24 N_MU=12 N_WALL=3   mesh resolution
#   CONFIG=config/lv_ellipsoid_em_video.options
#   WARP=1.0          displacement amplification for rendering
#   FPS=15            frames per second in the MP4

set -euo pipefail

NP=${NP:-4}
N_PHI=${N_PHI:-24}
N_MU=${N_MU:-12}
N_WALL=${N_WALL:-3}
CONFIG=${CONFIG:-config/lv_ellipsoid_em_video.options}
WARP=${WARP:-1.0}
FPS=${FPS:-15}

cd "$(dirname "$0")/.."

echo "[step 1/3] generating LV mesh"
mpirun --allow-run-as-root -np 1 ./build/generate_lv_ellipsoid_case \
  --out-dir benchmarks/lv_ellipsoid \
  --n-phi "$N_PHI" --n-mu "$N_MU" --n-wall "$N_WALL"

echo "[step 2/3] running monodomain + EM ($NP MPI ranks)"
rm -rf output/lv_ellipsoid_em_video
mpirun --allow-run-as-root -np "$NP" ./build/monodomain --config "$CONFIG"

echo "[step 3/3] rendering MP4"
python3.12 tools/render_lv_video.py \
  --case output/lv_ellipsoid_em_video \
  --warp "$WARP" \
  --fps "$FPS"

echo "done: output/lv_ellipsoid_em_video/lv_em.mp4"
