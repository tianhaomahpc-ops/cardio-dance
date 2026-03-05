#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_list(s: str) -> List[float]:
  out = []
  for token in s.split(","):
    t = token.strip()
    if not t:
      continue
    out.append(float(t))
  if not out:
    raise RuntimeError("empty candidate list")
  return out


def load_options(path: Path) -> List[str]:
  with path.open("r") as f:
    return [line.rstrip("\n") for line in f]


def write_options(
    lines: Iterable[str],
    path: Path,
    overrides: Dict[str, str],
    output_dir: str,
    checkpoint_dir: str,
) -> None:
  keys = set(overrides.keys()) | {"output_dir", "checkpoint_dir"}
  seen = set()
  out_lines: List[str] = []
  for raw in lines:
    s = raw.strip()
    if not s or s.startswith("#") or "=" not in s:
      out_lines.append(raw)
      continue
    key = s.split("=", 1)[0].strip()
    if key in overrides:
      out_lines.append(f"{key}={overrides[key]}")
      seen.add(key)
    elif key == "output_dir":
      out_lines.append(f"output_dir={output_dir}")
      seen.add("output_dir")
    elif key == "checkpoint_dir":
      out_lines.append(f"checkpoint_dir={checkpoint_dir}")
      seen.add("checkpoint_dir")
    else:
      out_lines.append(raw)

  for key, value in overrides.items():
    if key not in seen:
      out_lines.append(f"{key}={value}")
  if "output_dir" not in seen:
    out_lines.append(f"output_dir={output_dir}")
  if "checkpoint_dir" not in seen:
    out_lines.append(f"checkpoint_dir={checkpoint_dir}")

  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    f.write("\n".join(out_lines) + "\n")


def load_activation(path: Path) -> Dict[str, float]:
  with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames or []
    if "time_ms" not in cols:
      raise RuntimeError(f"Invalid probe csv (missing time_ms): {path}")
    probe_cols = [c for c in cols if c != "time_ms"]

    rows = list(reader)
    time = [float(r["time_ms"]) for r in rows]
    out: Dict[str, float] = {}
    for p in probe_cols:
      prev_t = time[0] if time else 0.0
      prev_v = float(rows[0][p]) if rows else float("nan")
      t_act = float("nan")
      for i in range(1, len(rows)):
        t = time[i]
        v = float(rows[i][p])
        if math.isfinite(prev_v) and math.isfinite(v) and prev_v < 0.0 <= v:
          dv = v - prev_v
          if abs(dv) < 1e-14:
            t_act = t
          else:
            t_act = prev_t + (-prev_v) * (t - prev_t) / dv
          break
        prev_t = t
        prev_v = v
      out[p] = t_act
    return out


def rmse(target: Dict[str, float], measured: Dict[str, float], miss_penalty_ms: float) -> float:
  errs = []
  for key, tgt in target.items():
    val = measured.get(key, float("nan"))
    if not math.isfinite(val):
      errs.append(miss_penalty_ms * miss_penalty_ms)
    else:
      d = val - tgt
      errs.append(d * d)
  if not errs:
    return float("inf")
  return math.sqrt(sum(errs) / len(errs))


def run_case(cmd: List[str]) -> int:
  print("[run]", " ".join(cmd))
  p = subprocess.run(cmd)
  return p.returncode


def main() -> None:
  ap = argparse.ArgumentParser(description="Grid-search calibration for Purkinje+AV-delay case")
  ap.add_argument("--base-config", default="config/niederer_purkinje_avdelay.options")
  ap.add_argument("--target-json", required=True, help="JSON: {probe_name: activation_ms}")
  ap.add_argument("--work-dir", default="output/calibration/purkinje_avdelay")
  ap.add_argument("--np", type=int, default=1)
  ap.add_argument("--exe", default="./build/monodomain")
  ap.add_argument("--mesh-tool", default="./build/split_niederer_x_regions_av")
  ap.add_argument("--mesh-in", default="benchmarks/niederer/niederer_benchmark.mesh")
  ap.add_argument("--mesh-out", default="benchmarks/niederer/niederer_purkinje_avdelay.mesh")
  ap.add_argument("--pvj-g-list", default="0.5,0.8,1.1")
  ap.add_argument("--purkinje-edge-g-list", default="1.2,1.6,2.0")
  ap.add_argument("--av-delay-scale-list", default="0.01,0.02,0.04")
  ap.add_argument("--fibrosis-scale-list", default="0.05,0.08,0.12")
  ap.add_argument("--miss-penalty-ms", type=float, default=15.0)
  args = ap.parse_args()

  base_config = Path(args.base_config)
  work_dir = Path(args.work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)

  with Path(args.target_json).open("r") as f:
    target = {k: float(v) for k, v in json.load(f).items()}

  # Ensure segmented mesh exists (idempotent).
  mesh_cmd = [
      "mpirun",
      "-np",
      "1",
      args.mesh_tool,
      "--in-mesh",
      args.mesh_in,
      "--out-mesh",
      args.mesh_out,
  ]
  if run_case(mesh_cmd) != 0:
    raise RuntimeError("mesh generation failed")

  base_lines = load_options(base_config)
  pvj_g_list = parse_list(args.pvj_g_list)
  edge_g_list = parse_list(args.purkinje_edge_g_list)
  av_scale_list = parse_list(args.av_delay_scale_list)
  fib_scale_list = parse_list(args.fibrosis_scale_list)

  combos: List[Tuple[float, float, float, float]] = list(
      itertools.product(pvj_g_list, edge_g_list, av_scale_list, fib_scale_list)
  )
  print(f"[calib] total runs: {len(combos)}")

  rows = []
  best = None
  for idx, (pvj_g, edge_g, av_s, fib_s) in enumerate(combos, start=1):
    tag = f"run_{idx:03d}"
    output_dir = work_dir / tag / "output"
    checkpoint_dir = work_dir / tag / "checkpoint"
    cfg_path = work_dir / tag / "case.options"

    overrides = {
        "enable_purkinje": "1",
        "pvj_g_mS": f"{pvj_g}",
        "purkinje_edge_g_mS": f"{edge_g}",
        "av_delay_sigma_scale": f"{av_s}",
        "fibrosis_sigma_scale": f"{fib_s}",
    }
    write_options(base_lines, cfg_path, overrides, str(output_dir), str(checkpoint_dir))

    cmd = [
        "mpirun",
        "-np",
        str(args.np),
        args.exe,
        "--config",
        str(cfg_path),
        "--benchmark-probes",
        "1",
    ]
    rc = run_case(cmd)
    probe_csv = output_dir / "probe_vm.csv"
    if rc != 0 or not probe_csv.exists():
      score = float("inf")
      activation = {}
    else:
      activation = load_activation(probe_csv)
      score = rmse(target, activation, args.miss_penalty_ms)

    row = {
        "tag": tag,
        "pvj_g_mS": pvj_g,
        "purkinje_edge_g_mS": edge_g,
        "av_delay_sigma_scale": av_s,
        "fibrosis_sigma_scale": fib_s,
        "rmse_ms": score,
    }
    rows.append(row)

    if best is None or score < best["rmse_ms"]:
      best = dict(row)
      best["activation"] = activation

    print(f"[calib] {tag} rmse_ms={score}")

  summary_csv = work_dir / "summary.csv"
  with summary_csv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "tag",
            "pvj_g_mS",
            "purkinje_edge_g_mS",
            "av_delay_sigma_scale",
            "fibrosis_sigma_scale",
            "rmse_ms",
        ],
    )
    w.writeheader()
    for r in rows:
      w.writerow(r)

  if best is None:
    raise RuntimeError("no calibration runs completed")

  best_json = work_dir / "best.json"
  with best_json.open("w") as f:
    json.dump(best, f, indent=2)

  print(f"[calib] summary: {summary_csv}")
  print(f"[calib] best: {best_json}")


if __name__ == "__main__":
  main()
