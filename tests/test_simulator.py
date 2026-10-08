import numpy as np
import pytest

from fleettwin.config import CityConfig, FleetConfig, OpsConfig, Policy
from fleettwin.plans import BASELINE, plan_arrays, plan_grid
from fleettwin.sim_reference import KPI_NAMES, simulate_reference
from fleettwin.world import build_world

SMALL_CITY = CityConfig(radius_rings=4, daily_requests=300.0)
SMALL_FLEET = FleetConfig(n_vehicles=60, initial_incidents=2, initial_workshop_faults=3)
TWO_SHIFTS = OpsConfig(shifts=((8.0, 4.0), (17.0, 4.0)), battery_capacity=6)

PLANS = [
    BASELINE,
    Policy(name="opp300_idle24_demand_soft", swap_opportunistic_permille=300,
           relocate_idle_hours=24.0, relocation_target=1, class_penalty_m=2_000),
    Policy(name="opp350_idle36_centre_soft", swap_opportunistic_permille=350,
           relocate_idle_hours=36.0, relocation_target=0, class_penalty_m=1_000),
    Policy(name="opp400_idle24_demand_strict", relocate_idle_hours=24.0, relocation_target=1,
           pair_radius_m=3000),
]


@pytest.fixture(scope="module")
def world():
    return build_world(SMALL_CITY, SMALL_FLEET, TWO_SHIFTS)


def reference_grid(world, plans, scenarios):
    arrays = plan_arrays(plans)
    res = {}
    for k in range(len(plans)):
        plan = {n: a[k] for n, a in arrays.items()}
        for j, m in enumerate(scenarios):
            res[k, j] = simulate_reference(world, plan, int(m))
    return res


def test_batched_matches_reference(world):
    torch = pytest.importorskip("torch")
    from fleettwin.sim_batched import simulate_batched

    scenarios = np.array([0, 1, 2, 3, 17, 42])
    ref = reference_grid(world, PLANS, scenarios)
    out = simulate_batched(world, plan_arrays(PLANS), scenarios, device="cpu", chunk_scenarios=4)
    for (k, j), r in ref.items():
        for n in KPI_NAMES:
            assert out["kpi"][n][k, j] == r["kpi"][n], (PLANS[k].name, int(scenarios[j]), n)
        for n, v in r["state"].items():
            np.testing.assert_array_equal(out["state"][n][k, j], v, err_msg=f"{n} k={k} j={j}")


def test_every_task_type_is_exercised(world):
    ref = reference_grid(world, PLANS, np.arange(6))
    total = {n: sum(r["kpi"][n] for r in ref.values()) for n in KPI_NAMES}
    for n in ("swaps", "repairs", "relocated", "recovered", "reloads"):
        assert total[n] > 0, n


def test_invariants_and_determinism(world):
    plan = {n: a[0] for n, a in plan_arrays([BASELINE]).items()}
    a = simulate_reference(world, plan, 5)
    b = simulate_reference(world, plan, 5)
    assert a["kpi"] == b["kpi"]
    k, s = a["kpi"], a["state"]
    assert k["served"] + k["lost"] == k["requests"]
    assert s["soc"].min() >= 0 and s["soc"].max() <= 1000
    assert s["cell"].min() >= 0 and s["cell"].max() < world.n_cells
    assert set(np.unique(s["fault"])) <= {0, 1, 2, 3}


def test_common_random_numbers_across_plans(world):
    """Demand depends only on the scenario, never on the plan."""
    plans = plan_grid()[:5]
    arrays = plan_arrays(plans)
    reqs = []
    for k in range(len(plans)):
        plan = {n: a[k] for n, a in arrays.items()}
        reqs.append(simulate_reference(world, plan, 9)["kpi"]["requests"])
    assert len(set(reqs)) == 1


def test_plan_grid():
    plans = plan_grid()
    assert len(plans) == 36
    assert plans[0].name == "baseline"
    assert len({p.name for p in plans}) == 36
