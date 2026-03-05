#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _to_float(v: str) -> float:
  try:
    return float(v)
  except Exception:
    return float("nan")


def first_zero_crossing(time_ms: List[float], vm: List[float]) -> Optional[float]:
  if len(time_ms) != len(vm) or len(vm) < 2:
    return None
  prev_t = time_ms[0]
  prev_v = vm[0]
  for i in range(1, len(vm)):
    t = time_ms[i]
    v = vm[i]
    if not (math.isfinite(prev_t) and math.isfinite(prev_v) and math.isfinite(t) and math.isfinite(v)):
      prev_t = t
      prev_v = v
      continue
    if prev_v < 0.0 <= v:
      dv = v - prev_v
      if abs(dv) < 1e-14:
        return t
      return prev_t + (-prev_v) * (t - prev_t) / dv
    prev_t = t
    prev_v = v
  return None


def main() -> None:
  ap = argparse.ArgumentParser(description="Extract first Vm zero-crossing activation times from probe_vm.csv")
  ap.add_argument("--csv", required=True, help="Path to probe_vm.csv")
  ap.add_argument(
      "--probes",
      default="Atria,AVDelay,Fibrosis,Ventricle",
      help="Comma-separated probe column names",
  )
  ap.add_argument("--out", default="", help="Optional JSON output path")
  args = ap.parse_args()

  csv_path = Path(args.csv)
  if not csv_path.exists():
    raise RuntimeError(f"CSV not found: {csv_path}")

  probes = [p.strip() for p in args.probes.split(",") if p.strip()]
  if not probes:
    raise RuntimeError("No probes specified")

  with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames or []
    if "time_ms" not in cols:
      raise RuntimeError("probe_vm.csv missing time_ms column")
    for p in probes:
      if p not in cols:
        raise RuntimeError(f"probe_vm.csv missing probe column: {p}")

    time_ms: List[float] = []
    probe_vm: Dict[str, List[float]] = {p: [] for p in probes}
    for row in reader:
      time_ms.append(_to_float(row.get("time_ms", "nan")))
      for p in probes:
        probe_vm[p].append(_to_float(row.get(p, "nan")))

  activation = {}
  for p in probes:
    t_act = first_zero_crossing(time_ms, probe_vm[p])
    activation[p] = t_act

  for p in probes:
    v = activation[p]
    if v is None:
      print(f"{p}: nan")
    else:
      print(f"{p}: {v:.6f} ms")

  if args.out:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
      json.dump(activation, f, indent=2)


if __name__ == "__main__":
  main()
