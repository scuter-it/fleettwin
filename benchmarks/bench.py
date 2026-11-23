"""Throughput benchmark: reference simulator vs batched simulator.

Examples
    python benchmarks/bench.py --backend reference --plans 36 --scenarios 32
    python benchmarks/bench.py --backend torch --device cuda --plans 36 --scenarios 64,256,1024
    nsys profile -t cuda,nvtx -o fleettwin python benchmarks/bench.py --backend torch \
        --device cuda --plans 36 --scenarios 256 --repeat 1

The reference runs one (plan, scenario) pair at a time, so its time grows
linearly with K x M; --reference-pairs caps how many pairs are timed and the
result is reported per pair. Every run appends one JSON line to
results/bench.jsonl with the hardware, sizes and timings.
"""

import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleettwin.config import CityConfig, FleetConfig  # noqa: E402
from fleettwin.plans import plan_arrays, plan_grid, tile_plans  # noqa: E402
from fleettwin.sim_reference import simulate_reference  # noqa: E402
from fleettwin.world import build_world  # noqa: E402


def bench_reference(world, plans, scen, max_pairs):
    arrays = plan_arrays(plans)
    pairs = [(k, int(m)) for k in range(len(plans)) for m in scen][:max_pairs]
    t0 = time.perf_counter()
    for k, m in pairs:
        simulate_reference(world, {n: a[k] for n, a in arrays.items()}, m)
    return time.perf_counter() - t0, len(pairs)


def bench_torch(world, plans, scen, device, chunk, repeat):
    from fleettwin.sim_batched import DeviceTables, simulate_batched

    tables = DeviceTables(world, device)
    arrays = plan_arrays(plans)
    simulate_batched(world, arrays, scen[:2], device=device, tables=tables)  # warm-up
    times = []
    for _ in range(repeat):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        simulate_batched(world, arrays, scen, device=device, chunk_scenarios=chunk, tables=tables)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times), len(plans) * len(scen)


def hardware(device):
    info = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    if str(device).startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["torch"] = torch.__version__
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["reference", "torch"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plans", type=int, default=36)
    ap.add_argument("--scenarios", default="64", help="comma-separated list of M values")
    ap.add_argument("--vehicles", type=int, default=FleetConfig.n_vehicles)
    ap.add_argument("--rings", type=int, default=CityConfig.radius_rings)
    ap.add_argument("--chunk", type=int, default=None, help="scenarios per chunk (memory bound)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--reference-pairs", type=int, default=64)
    ap.add_argument("--out", default="results/bench.jsonl")
    args = ap.parse_args()

    world = build_world(CityConfig(radius_rings=args.rings), FleetConfig(n_vehicles=args.vehicles))
    plans = tile_plans(plan_grid(), args.plans)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for M in [int(x) for x in args.scenarios.split(",")]:
        scen = torch.arange(M)
        if args.backend == "reference":
            secs, pairs = bench_reference(world, plans, scen, args.reference_pairs)
        else:
            secs, pairs = bench_torch(world, plans, scen, args.device, args.chunk, args.repeat)
        rec = {
            "backend": args.backend, "device": args.device, "plans": args.plans, "scenarios": M,
            "vehicles": world.n_vehicles, "cells": world.n_cells, "steps": world.n_steps,
            "pairs_timed": pairs, "seconds": secs,
            "seconds_per_pair": secs / pairs,
            "pairs_per_second": pairs / secs,
            "vehicle_steps_per_second": pairs * world.n_vehicles * world.n_steps / secs,
            "hardware": hardware(args.device),
        }
        print(json.dumps(rec))
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
