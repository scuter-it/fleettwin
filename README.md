# FleetTwin

FleetTwin scores operational plans for a shared light-vehicle fleet under demand uncertainty. It simulates K candidate plans against M sampled demand scenarios over a 24-hour horizon and reports, for each plan, the mean and 10th–90th percentiles of operational KPIs.

The simulator is parametric, not learned. The repository contains a readable reference simulator and a batched one that must reproduce it exactly. Both are written in PyTorch and run on CPU or GPU. All data is synthetic; nothing here depends on proprietary data or code.

Status: v0.1, first working version. It has not yet been optimised or run at scale on GPUs.

## What it models

A single operator with one van works in shifts. Tasks follow today's single-pass priority heuristic:

1. **Incidents.** Damaged vehicles are recovered to the workshop. The van has two slots, so the nearest other vehicle needing workshop work is collected on the same trip.
2. **Battery swaps.** Mandatory below a threshold (default 30%). At each stop, nearby vehicles below a second threshold (default 40%) are swapped as well, while batteries last. The van reloads at the depot when empty.
3. **On-site repairs.** Flat tyres, broken parts, cleaning.
4. **Relocation.** Vehicles idle for longer than a threshold (default 48 h) and outside the centre are moved, two at a time, towards the centre or towards the cells with the largest forecast demand deficit.

Within a priority class the operator takes the nearest task. A softer variant trades class against distance. Plans are variants of these thresholds and choices; `fleettwin.plans.plan_grid()` builds 36 of them, with the baseline first.

Demand is Poisson per H3 cell and 15-minute step, with scenario-level demand levels and an optional event spike at a venue. Trip destinations follow a gravity-style origin-destination model. Rentals match requests to available vehicles in the same cell, most-charged first. See [docs/model.md](docs/model.md) for the full model and its simplifications.

## Design choices for GPU execution

- **State layout.** One integer tensor per attribute, shape `[K*M, V]`: cell, state of charge (per mille), idle steps, fault level, busy-until step.
- **Counter-based random numbers.** Every draw is a hash of (seed, stream, scenario, step, cell or vehicle). Draws need no stored state, are identical on CPU and GPU, and are shared by all plans for the same scenario (common random numbers).
- **Integer arithmetic throughout.** Distances in metres, times in seconds, probabilities as thresholds on 32-bit uniforms. The batched result equals the reference bit for bit, so correctness is tested exactly, not to a tolerance.
- **Independent scenario chunks.** Scenarios are independent and can be processed in chunks. The same split is the basis for distributing work across GPUs.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # equivalence and invariant tests
python -m fleettwin.cli --backend reference --scenarios 16
python -m fleettwin.cli --backend torch --device cuda --scenarios 1024
```

## Docker

The default image uses PyTorch with CUDA 13 and runs on NVIDIA GPUs, including Jetson with JetPack 7. The CPU-only image is smaller.

```bash
docker build -t fleettwin .
docker build -t fleettwin --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .   # CPU only
docker run --rm --gpus all fleettwin python -m pytest -q
docker run --rm --gpus all fleettwin python -m fleettwin.cli --device cuda --scenarios 1024
```

## Benchmarks

```bash
python benchmarks/bench.py --backend reference --plans 36 --scenarios 32
python benchmarks/bench.py --backend torch --device cuda --plans 36 --scenarios 64,256,1024
```

Each run appends a JSON line to `results/bench.jsonl`, recording sizes, timings and hardware. Profile the batched simulator with Nsight Systems; the step loop is annotated with NVTX ranges:

```bash
nsys profile -t cuda,nvtx -o fleettwin python benchmarks/bench.py \
    --backend torch --device cuda --plans 36 --scenarios 256 --repeat 1
```

| Backend | Hardware | Vehicles | Time per (plan, scenario) |
| --- | --- | --- | --- |
| Reference (v0.1, NumPy) | 1 vCPU, x86_64 | 200 | 22.5 ms |
| Reference (v0.1, NumPy) | 1 vCPU, x86_64 | 1,000 | 41.5 ms |
| PyTorch batched | GPU | 200 | to be measured |

## Known limitations of v0.1

- The time loop runs in Python, with many small kernels per step. Removing this overhead is the first optimisation target.
- Reductions over scenarios run on the host.
- Operator routing is greedy, nearest task first.
- Every parameter in `fleettwin/config.py` marked PLACEHOLDER is synthetic.
- No public dataset loader and no demand forecaster yet; demand rates are synthetic.

## Licence

Apache License 2.0; see [LICENSE](LICENSE). Copyright 2026 Scuter S.r.l.
