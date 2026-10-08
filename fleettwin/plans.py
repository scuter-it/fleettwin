"""Candidate plans.

The baseline encodes today's single-pass priority heuristic:
incidents first, then mandatory battery swaps (opportunistic swaps near
each stop), then on-site repairs, then relocation of vehicles idle for
48 hours towards the centre. Variants change one or more thresholds.
"""

from itertools import product

import numpy as np

from .config import Policy, STEP_SECONDS

BASELINE = Policy(name="baseline")


def plan_grid() -> list:
    """36 plans: baseline first, then every combination of the four rules."""
    plans = [BASELINE]
    for opp, idle_h, target, pen in product((300, 350, 400), (48.0, 36.0, 24.0), (0, 1),
                                            (10_000_000, 2_000)):
        p = Policy(
            name=f"opp{opp}_idle{int(idle_h)}h_{'centre' if target == 0 else 'demand'}"
                 f"_{'strict' if pen > 100_000 else 'soft'}",
            swap_opportunistic_permille=opp,
            relocate_idle_hours=idle_h,
            relocation_target=target,
            class_penalty_m=pen,
        )
        if (p.swap_opportunistic_permille, p.relocate_idle_hours, p.relocation_target,
                p.class_penalty_m) == (BASELINE.swap_opportunistic_permille,
                                       BASELINE.relocate_idle_hours,
                                       BASELINE.relocation_target,
                                       BASELINE.class_penalty_m):
            continue
        plans.append(p)
    return plans


def tile_plans(plans: list, k: int) -> list:
    """Repeat a plan list up to k entries (for benchmarks with arbitrary K)."""
    return [plans[i % len(plans)] for i in range(k)]


def plan_arrays(plans: list) -> dict:
    """Per-plan parameters as int64 arrays of length K."""
    to_steps = lambda h: int(round(h * 3600 / STEP_SECONDS))
    return {
        "mand": np.array([p.swap_mandatory_permille for p in plans], dtype=np.int64),
        "opp": np.array([p.swap_opportunistic_permille for p in plans], dtype=np.int64),
        "opp_r": np.array([p.opportunistic_radius_m for p in plans], dtype=np.int64),
        "idle": np.array([to_steps(p.relocate_idle_hours) for p in plans], dtype=np.int64),
        "target": np.array([p.relocation_target for p in plans], dtype=np.int64),
        "pair_r": np.array([p.pair_radius_m for p in plans], dtype=np.int64),
        "pen": np.array([p.class_penalty_m for p in plans], dtype=np.int64),
    }
