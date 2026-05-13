#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
  parser = argparse.ArgumentParser(description="Plot regional Vm probes from probe_vm.csv")
  parser.add_argument("--csv", required=True, help="Path to probe_vm.csv")
  parser.add_argument("--out", required=True, help="Output png path")
  parser.add_argument("--atria-col", default="Atria", help="CSV column for atria probe")
  parser.add_argument("--fibrosis-col", default="Fibrosis", help="CSV column for fibrosis probe")
  parser.add_argument("--ventricle-col", default="Ventricle", help="CSV column for ventricle probe")
  args = parser.parse_args()

  csv_path = Path(args.csv)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(csv_path)
  required = ["time_ms", args.atria_col, args.fibrosis_col, args.ventricle_col]
  missing = [c for c in required if c not in df.columns]
  if missing:
    raise RuntimeError(f"Missing columns in CSV: {missing}")

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.plot(df["time_ms"], df[args.atria_col], label=f"{args.atria_col} Vm")
  ax.plot(df["time_ms"], df[args.fibrosis_col], label=f"{args.fibrosis_col} Vm")
  ax.plot(df["time_ms"], df[args.ventricle_col], label=f"{args.ventricle_col} Vm")
  ax.set_xlabel("Time (ms)")
  ax.set_ylabel("Vm (mV)")
  ax.set_title("Vm time series (regional probes)")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  print(f"Saved figure: {out_path}")


if __name__ == "__main__":
  main()
