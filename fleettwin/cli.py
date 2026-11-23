"""Compare candidate plans on a synthetic city.

    python -m fleettwin.cli --scenarios 128 --backend torch --device cuda
"""

import argparse
import time

import torch

from .config import CityConfig, FleetConfig
from .kpi import format_table, summarise
from .plans import BASELINE, plan_arrays, plan_grid
from .sim_reference import KPI_NAMES, simulate_reference
from .world import build_world


def run_reference(world, plans_list, scenario_ids):
    arrays = plan_arrays(plans_list)
    K, M = len(plans_list), len(scenario_ids)
    out = {n: torch.zeros((K, M), dtype=torch.int64) for n in KPI_NAMES}
    for k in range(K):
        plan = {n: a[k] for n, a in arrays.items()}
        for j, m in enumerate(scenario_ids):
            res = simulate_reference(world, plan, int(m))
            for n in KPI_NAMES:
                out[n][k, j] = res["kpi"][n]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", choices=["grid", "baseline"], default="grid")
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--backend", choices=["reference", "torch"], default="torch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vehicles", type=int, default=FleetConfig.n_vehicles)
    ap.add_argument("--rings", type=int, default=CityConfig.radius_rings)
    ap.add_argument("--top", type=int, default=10, help="plans to print, by vehicles moved per shift hour")
    args = ap.parse_args(argv)

    world = build_world(CityConfig(radius_rings=args.rings), FleetConfig(n_vehicles=args.vehicles))
    plans_list = plan_grid() if args.plans == "grid" else [BASELINE]
    scen = torch.arange(args.scenarios)

    t0 = time.perf_counter()
    if args.backend == "reference":
        kpi = run_reference(world, plans_list, scen)
    else:
        from .sim_batched import simulate_batched
        kpi = simulate_batched(world, plan_arrays(plans_list), scen, device=args.device)["kpi"]
    elapsed = time.perf_counter() - t0

    rows = summarise(kpi, world, [p.name for p in plans_list])
    base, rest = rows[0], rows[1:]
    rest.sort(key=lambda r: -r["moved_per_shift_h"][0])
    print(f"{world.n_cells} cells, {world.n_vehicles} vehicles, {len(plans_list)} plans x "
          f"{len(scen)} scenarios, {args.backend} on {args.device}: {elapsed:.2f} s")
    print("Values: mean [p10-p90] over scenarios. Synthetic city: not operational advice.\n")
    print(format_table([base] + rest[:args.top]))


if __name__ == "__main__":
    main()
