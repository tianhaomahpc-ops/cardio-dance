#!/usr/bin/env python3
"""Generate a Costabal-style fractal Purkinje network.

The output `.network` file is consumable by PurkinjeCableSolver. The algorithm:
- Starts from `--root` and grows in a user-supplied initial direction.
- Each step: extend by `--length` mm with small Gaussian jitter, then
  bifurcate with probability `--p_bifurcate` (until `--max_depth` is reached).
- Branch angles are drawn from N(`--bifurcation_angle_deg`, `--angle_sigma`).
- Leaf nodes are written as terminals (terminal=1).
- An optional axis-aligned bounding box `--bbox_min`/`--bbox_max` clips growth
  to remain inside the heart endocardial volume.

Usage example:
  python tools/generate_purkinje_tree.py \
      --root 24,7,3 --direction -1,0,0 --length 2.5 --max_depth 8 \
      --bifurcation_angle_deg 30 --p_bifurcate 0.7 \
      --bbox_min 0,0,0 --bbox_max 30,7,3 \
      --out config/purkinje_v.network --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass


def parse_vec(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated floats, got {s!r}")
    return parts[0], parts[1], parts[2]


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    if n == 0:
        return (1.0, 0.0, 0.0)
    return tuple(c / n for c in v)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def rotate_about_axis(v, axis, theta_rad):
    # Rodrigues' rotation
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    ax, ay, az = axis
    vx, vy, vz = v
    dot = ax * vx + ay * vy + az * vz
    return (
        vx * cos_t + (ay * vz - az * vy) * sin_t + ax * dot * (1 - cos_t),
        vy * cos_t + (az * vx - ax * vz) * sin_t + ay * dot * (1 - cos_t),
        vz * cos_t + (ax * vy - ay * vx) * sin_t + az * dot * (1 - cos_t),
    )


@dataclass
class Node:
    x: float
    y: float
    z: float
    terminal: bool = False


def in_box(p, bbox_min, bbox_max):
    if bbox_min is None or bbox_max is None:
        return True
    return all(bbox_min[i] <= p[i] <= bbox_max[i] for i in range(3))


def generate(args):
    rng = random.Random(args.seed)

    nodes: list[Node] = []
    edges: list[tuple[int, int]] = []

    # Multiple roots are supported by repeating --root and --direction;
    # each pair starts an independent subtree that grows from its root.
    # Subtree node indices are concatenated into a single .network file
    # (no edges between subtrees -- they only meet at the heart side
    # through PVJ injection).
    roots_raw = args.root if isinstance(args.root, list) else [args.root]
    if not args.direction:
        dirs_raw = ["1,0,0"] * len(roots_raw)
    elif isinstance(args.direction, list):
        dirs_raw = args.direction
    else:
        dirs_raw = [args.direction]
    if len(dirs_raw) == 1 and len(roots_raw) > 1:
        dirs_raw = dirs_raw * len(roots_raw)
    if len(roots_raw) != len(dirs_raw):
        raise ValueError(
            f"--root count {len(roots_raw)} != --direction count {len(dirs_raw)}"
        )
    root_indices = []  # for caller / stim wiring -- prints node index of each root.
    bbox_min = parse_vec(args.bbox_min) if args.bbox_min else None
    bbox_max = parse_vec(args.bbox_max) if args.bbox_max else None

    angle_main_rad = math.radians(args.bifurcation_angle_deg)
    angle_sigma_rad = math.radians(args.angle_sigma)

    # Stack accumulates work across all subtrees so they share one rng stream;
    # we push each root's initial work-item up front.
    stack = []
    for root_str, dir_str in zip(roots_raw, dirs_raw):
        root = parse_vec(root_str)
        init_dir = normalize(parse_vec(dir_str))
        nodes.append(Node(*root))
        root_indices.append(len(nodes) - 1)
        stack.append((len(nodes) - 1, init_dir, 0))

    while stack:
        parent_idx, d, depth = stack.pop()
        if depth >= args.max_depth:
            nodes[parent_idx].terminal = True
            continue
        # Extend one segment.
        seg = args.length * (1.0 + rng.gauss(0.0, 0.1))
        seg = max(seg, 0.5 * args.length)
        new_pos = tuple(nodes[parent_idx].__dict__[c] + d[i] * seg
                        for i, c in enumerate(("x", "y", "z")))
        if not in_box(new_pos, bbox_min, bbox_max):
            nodes[parent_idx].terminal = True
            continue
        nodes.append(Node(*new_pos))
        new_idx = len(nodes) - 1
        edges.append((parent_idx, new_idx))

        # Optional bifurcation.
        if rng.random() < args.p_bifurcate:
            # Choose a perpendicular rotation axis with random orientation
            # around d. Without randomisation the axis was always cross(d, +z)
            # which collapses to (0, 1, 0) when d ~ ±x, trapping the entire
            # tree in the x-z plane and giving zero y-spread. We construct a
            # random unit vector in the plane perpendicular to d.
            #   1. Build any vector not parallel to d.
            #   2. Project out the d-component, normalise -> first basis e1.
            #   3. Cross d x e1 -> second basis e2.
            #   4. Mix e1, e2 with a random angle phi in [0, 2 pi) for the
            #      rotation axis.
            if abs(d[0]) > 0.9:
                tmp = (0.0, 1.0, 0.0)
            else:
                tmp = (1.0, 0.0, 0.0)
            dot_dt = d[0]*tmp[0] + d[1]*tmp[1] + d[2]*tmp[2]
            e1 = (tmp[0] - dot_dt*d[0],
                  tmp[1] - dot_dt*d[1],
                  tmp[2] - dot_dt*d[2])
            e1 = normalize(e1)
            e2 = cross(d, e1)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            axis = (math.cos(phi)*e1[0] + math.sin(phi)*e2[0],
                    math.cos(phi)*e1[1] + math.sin(phi)*e2[1],
                    math.cos(phi)*e1[2] + math.sin(phi)*e2[2])
            axis = normalize(axis)
            theta_l = angle_main_rad + rng.gauss(0.0, angle_sigma_rad)
            theta_r = -angle_main_rad + rng.gauss(0.0, angle_sigma_rad)
            d_left = normalize(rotate_about_axis(d, axis, theta_l))
            d_right = normalize(rotate_about_axis(d, axis, theta_r))
            stack.append((new_idx, d_left, depth + 1))
            stack.append((new_idx, d_right, depth + 1))
        else:
            # Continue growing same direction with small drift in a random
            # perpendicular direction (same logic as bifurcation axis).
            if abs(d[0]) > 0.9:
                tmp = (0.0, 1.0, 0.0)
            else:
                tmp = (1.0, 0.0, 0.0)
            dot_dt = d[0]*tmp[0] + d[1]*tmp[1] + d[2]*tmp[2]
            e1 = (tmp[0] - dot_dt*d[0],
                  tmp[1] - dot_dt*d[1],
                  tmp[2] - dot_dt*d[2])
            e1 = normalize(e1)
            e2 = cross(d, e1)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            axis = (math.cos(phi)*e1[0] + math.sin(phi)*e2[0],
                    math.cos(phi)*e1[1] + math.sin(phi)*e2[1],
                    math.cos(phi)*e1[2] + math.sin(phi)*e2[2])
            axis = normalize(axis)
            theta = rng.gauss(0.0, angle_sigma_rad)
            d_next = normalize(rotate_about_axis(d, axis, theta))
            stack.append((new_idx, d_next, depth + 1))

    # Leaves end up as terminals if not already flagged.
    incoming = [0] * len(nodes)
    for a, b in edges:
        incoming[b] += 1
    out_count = [0] * len(nodes)
    for a, _ in edges:
        out_count[a] += 1
    for i, n in enumerate(nodes):
        if out_count[i] == 0:
            n.terminal = True

    return nodes, edges, root_indices


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, action="append",
                   help="x,y,z root coordinate. Repeat for multiple roots.")
    p.add_argument("--direction", action="append",
                   help="initial growth direction. Repeat to match --root, or "
                        "supply once and it will be reused for every root.")
    p.add_argument("--length", type=float, default=2.5, help="segment length (mm)")
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--bifurcation_angle_deg", type=float, default=30.0)
    p.add_argument("--angle_sigma", type=float, default=8.0)
    p.add_argument("--p_bifurcate", type=float, default=0.7)
    p.add_argument("--bbox_min", default=None, help="x,y,z bounding-box min in mm")
    p.add_argument("--bbox_max", default=None, help="x,y,z bounding-box max in mm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output .network path")
    args = p.parse_args()

    nodes, edges, root_indices = generate(args)
    with open(args.out, "w") as f:
        f.write(f"{len(nodes)} {len(edges)}\n")
        for n in nodes:
            f.write(f"{n.x} {n.y} {n.z} {1 if n.terminal else 0}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")
    n_term = sum(1 for n in nodes if n.terminal)
    print(f"wrote {args.out}: {len(nodes)} nodes ({n_term} terminals), {len(edges)} edges",
          file=sys.stderr)
    if len(root_indices) > 1:
        print(
            f"  root node indices (use these in purkinje_stim_nodes): "
            f"{','.join(str(i) for i in root_indices)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
