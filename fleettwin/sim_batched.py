"""Batched PyTorch simulator: K plans x M scenarios in one tensor program.

Every state tensor has shape [B, V] with B = K * M (plan-major:
b = k * M + m). The semantics match sim_reference.py exactly; the
equivalence test checks every KPI and the final state.

The code is deliberately unoptimised (v0.1): one Python loop over 96
steps and, inside shifts, a bounded loop over operator jobs, each
launching many small kernels. Profiling and removing this overhead is the
work planned for the hackathon.
"""

from contextlib import contextmanager

import numpy as np
import torch

from . import rng
from .sim_reference import KPI_NAMES
from .world import OD_SCALE, World

INF = 1 << 62


@contextmanager
def nvtx(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


class DeviceTables:
    """World tables moved once to the target device."""

    def __init__(self, world: World, device):
        t = lambda a, dt=torch.int64: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.dist = t(world.dist_m)
        self.trav = t(world.travel_s)
        self.trip = t(world.trip_permille)
        self.od = t(world.od_flat)
        self.pthr = t(world.poisson_thr)
        self.fc = t(world.forecast_milli)
        self.in_centre = t(world.in_centre, torch.bool)
        self.init = {k: t(v) for k, v in world.init.items()}


def simulate_batched(world: World, plans: dict, scenario_ids, device="cpu",
                     chunk_scenarios: int | None = None, tables: DeviceTables | None = None) -> dict:
    """Simulate every plan in `plans` (from plan_arrays) under every scenario id.

    Returns KPI arrays of shape [K, M] (NumPy int64). Scenarios can be processed
    in chunks to bound memory; chunks are independent, which is also how the
    work will be split across GPUs.
    """
    scenario_ids = np.asarray(scenario_ids, dtype=np.int64)
    device = torch.device(device)
    tables = tables or DeviceTables(world, device)
    chunk = chunk_scenarios or len(scenario_ids)
    parts = []
    for i in range(0, len(scenario_ids), chunk):
        parts.append(_simulate_chunk(world, plans, scenario_ids[i:i + chunk], device, tables))
    kpi = {n: np.concatenate([p["kpi"][n] for p in parts], axis=1) for n in KPI_NAMES}
    state = {n: np.concatenate([p["state"][n] for p in parts], axis=1) for n in parts[0]["state"]}
    return {"kpi": kpi, "state": state}


def _simulate_chunk(world, plans, scen, device, tb: DeviceTables) -> dict:
    C, V, T, S = world.n_cells, world.n_vehicles, world.n_steps, world.step_seconds
    ops, fl = world.ops, world.fleet
    K, M = len(plans["mand"]), len(scen)
    B = K * M
    i64 = dict(dtype=torch.int64, device=device)
    use_nvtx = device.type == "cuda"
    seed = world.scenario_seed
    t3, t2, t1 = world.fault_thr
    workshop_steps = int(round(ops.workshop_repair_hours * 3600 / S))
    min_rent = fl.min_rentable_permille

    # Per-row plan parameters, [B, 1] for broadcasting over vehicles.
    P = {n: torch.as_tensor(a, **i64).repeat_interleave(M)[:, None] for n, a in plans.items()}
    m_t = torch.as_tensor(scen, **i64)[:, None]                        # [M, 1]
    variant = torch.as_tensor(world.scenario_variants(scen), **i64)   # [M]

    vidx = torch.arange(V, **i64)[None, :]
    cidx = torch.arange(C, **i64)[None, :]

    def rep(x):  # [M, ...] scenario-level tensor -> [B, ...]
        return x.repeat(K, *([1] * (x.dim() - 1)))

    def at(table, a, b):  # table lookup on flattened [C*C]
        return table[a * C + b]

    def pick(x, idx):  # x[b, idx[b]]
        return x.gather(1, idx[:, None]).squeeze(1)

    cell = tb.init["cell"][None, :].repeat(B, 1)
    soc = tb.init["soc"][None, :].repeat(B, 1)
    idle = tb.init["idle"][None, :].repeat(B, 1)
    fault = tb.init["fault"][None, :].repeat(B, 1)
    busy = tb.init["busy"][None, :].repeat(B, 1)

    op_cell = torch.full((B,), world.depot, **i64)
    op_free = torch.zeros(B, **i64)
    batteries = torch.full((B,), ops.battery_capacity, **i64)
    k = {n: torch.zeros(B, **i64) for n in KPI_NAMES}
    arange_v = torch.arange(V, **i64).repeat(B, 1)

    for t in range(T):
        step_start, step_end = t * S, (t + 1) * S
        if world.shift_start[t]:
            op_cell.fill_(world.depot)
            batteries.fill_(ops.battery_capacity)
            op_free = torch.clamp(op_free, min=step_start)

        # ---------------- field operations ----------------
        if world.shift_on[t]:
            with nvtx(f"ops_t{t}", use_nvtx):
                for _ in range(ops.max_jobs_per_step):
                    active = op_free < step_end
                    if not bool(active.any()):
                        break
                    start = torch.clamp(op_free, min=step_start)
                    free = busy <= t
                    d_op = at(tb.dist, op_cell[:, None], cell)
                    below_mand = soc < P["mand"]
                    c0 = free & (fault == 3)
                    c1 = free & (fault == 0) & below_mand
                    c2 = free & (fault == 1)
                    c3 = (free & (fault == 0) & ~below_mand & (idle >= P["idle"])
                          & ~tb.in_centre[cell])
                    cls = torch.full_like(cell, 4)
                    cls = torch.where(c3, 3, cls)
                    cls = torch.where(c2, 2, cls)
                    cls = torch.where(c1, 1, cls)
                    cls = torch.where(c0, 0, cls)
                    key = torch.where(cls < 4, (cls * P["pen"] + d_op) * V + vidx, INF)
                    best, p = key.min(dim=1)
                    has_job = active & (best < INF)
                    op_free = torch.where(active & ~has_job, step_end, op_free)

                    p_cls = pick(cls, p)
                    p_cell = pick(cell, p)
                    is_inc = has_job & (p_cls == 0)
                    is_rel = has_job & (p_cls == 3)
                    is_reload = has_job & (p_cls == 1) & (batteries == 0)
                    is_swap = has_job & (p_cls == 1) & (batteries > 0)
                    is_rep = has_job & (p_cls == 2)
                    is_pick = is_inc | is_rel
                    d_p = at(tb.dist, p_cell[:, None], cell)
                    is_p = vidx == p[:, None]

                    # Partner for the two-slot van.
                    elig_q = (torch.where(is_inc[:, None], free & (fault >= 2), c3) & ~is_p
                              & (d_p <= P["pair_r"]) & is_pick[:, None])
                    key_q = torch.where(elig_q, d_p * V + vidx, INF)
                    best_q, q = key_q.min(dim=1)
                    has_q = is_pick & (best_q < INF)
                    q_cell = pick(cell, q)

                    # Relocation target from current availability.
                    avail = ((fault == 0) | (fault == 2)) & (soc >= min_rent) & free
                    counts = torch.zeros(B, C + 1, **i64).scatter_add_(
                        1, torch.where(avail, cell, C), torch.ones_like(cell))[:, :C]
                    key_c = torch.where(tb.in_centre[None, :], counts * C + cidx, INF)
                    key_d = (10**9 - (tb.fc[t][None, :] - 1000 * counts)) * C + cidx
                    target = torch.where(P["target"][:, 0] == 0, key_c.argmin(1), key_d.argmin(1))
                    drop = torch.where(is_inc, world.workshop, target)
                    last = torch.where(has_q, q_cell, p_cell)
                    hq = has_q.long()

                    dur_pick = (at(tb.trav, op_cell, p_cell) + ops.pickup_seconds
                                + hq * (at(tb.trav, p_cell, q_cell) + ops.pickup_seconds)
                                + at(tb.trav, last, drop) + ops.drop_seconds * (1 + hq))
                    m_pick = (at(tb.dist, op_cell, p_cell) + hq * at(tb.dist, p_cell, q_cell)
                              + at(tb.dist, last, drop))

                    # Battery swap with opportunistic neighbours.
                    elig_s = (free & (fault == 0) & (soc < P["opp"]) & (d_p <= P["opp_r"])
                              & is_swap[:, None])
                    key_s = torch.where(elig_s, torch.where(is_p, -1, d_p * V + vidx), INF)
                    order = torch.sort(key_s, dim=1).indices
                    rank = torch.empty_like(order).scatter_(1, order, arange_v)
                    sel = elig_s & (rank < batteries[:, None])
                    n_sel = sel.sum(1)
                    tr_p = at(tb.trav, p_cell[:, None], cell)
                    dur_swap = (at(tb.trav, op_cell, p_cell) + torch.where(sel, tr_p, 0).sum(1)
                                + ops.swap_seconds * n_sel)
                    m_swap = at(tb.dist, op_cell, p_cell) + torch.where(sel, d_p, 0).sum(1)

                    dur_reload = at(tb.trav, op_cell, world.depot) + ops.reload_seconds
                    m_reload = at(tb.dist, op_cell, world.depot)
                    dur_rep = at(tb.trav, op_cell, p_cell) + ops.repair_onsite_seconds
                    m_rep = at(tb.dist, op_cell, p_cell)

                    zero = torch.zeros_like(op_free)
                    dur = torch.where(is_pick, dur_pick, torch.where(is_swap, dur_swap,
                          torch.where(is_reload, dur_reload, torch.where(is_rep, dur_rep, zero))))
                    meters = torch.where(is_pick, m_pick, torch.where(is_swap, m_swap,
                             torch.where(is_reload, m_reload, torch.where(is_rep, m_rep, zero))))
                    end = start + dur
                    end_step = ((end + S - 1) // S)[:, None]

                    # Effects on vehicles.
                    moved = is_pick[:, None] & (is_p | (has_q[:, None] & (vidx == q[:, None])))
                    cell = torch.where(moved, drop[:, None], cell)
                    idle = torch.where(moved, 0, idle)
                    fault = torch.where(moved & is_inc[:, None], 0, fault)
                    busy = torch.where(moved, torch.where(is_inc[:, None],
                                                          end_step + workshop_steps, end_step), busy)
                    soc = torch.where(sel, 1000, soc)
                    busy = torch.where(sel, end_step, busy)
                    rep_v = is_rep[:, None] & is_p
                    fault = torch.where(rep_v, 0, fault)
                    busy = torch.where(rep_v, end_step, busy)

                    # Effects on the operator.
                    batteries = torch.where(is_swap, batteries - n_sel, batteries)
                    batteries = torch.where(is_reload, ops.battery_capacity, batteries)
                    op_cell = torch.where(is_pick, drop, torch.where(is_swap | is_rep, p_cell,
                              torch.where(is_reload, world.depot, op_cell)))
                    op_free = torch.where(has_job, end, op_free)

                    n_moved = 1 + hq
                    k["recovered"] += torch.where(is_inc, n_moved, 0)
                    k["relocated"] += torch.where(is_rel, n_moved, 0)
                    k["reloads"] += is_reload.long()
                    k["swaps"] += torch.where(is_swap, n_sel, 0)
                    k["repairs"] += is_rep.long()
                    k["jobs"] += has_job.long()
                    k["op_m"] += torch.where(has_job, meters, 0)

        # ---------------- rentals ----------------
        with nvtx(f"rent_t{t}", use_nvtx):
            avail = ((fault == 0) | (fault == 2)) & (soc >= min_rent) & (busy <= t)
            u = rng.counter_hash(seed, rng.DEMAND, m_t, t, cidx)          # [M, C]
            thr = tb.pthr[variant, t]                                       # [M, C, KMAX]
            demand = rep((u[:, :, None] >= thr).sum(2))                     # [B, C]
            cellkey = torch.where(avail, cell, C)
            key = cellkey * (1001 * V) + (1000 - soc) * V + vidx
            skey, order = torch.sort(key, dim=1)
            scell = skey // (1001 * V)
            is_start = torch.ones_like(scell, dtype=torch.bool)
            is_start[:, 1:] = scell[:, 1:] != scell[:, :-1]
            seg_start = torch.cummax(torch.where(is_start, arange_v, 0), dim=1).values
            rank = torch.empty_like(order).scatter_(1, order, arange_v - seg_start)
            demand_ext = torch.cat([demand, torch.zeros(B, 1, **i64)], dim=1)
            rented = avail & (rank < demand_ext.gather(1, cellkey))

            u_dest = rep(rng.counter_hash(seed, rng.DEST, m_t, t, vidx) >> 2)
            dest = torch.searchsorted(tb.od, cell * OD_SCALE + u_dest, right=True) - cell * C
            cons = at(tb.trip, cell, dest)
            uf = rep(rng.counter_hash(seed, rng.FAULT, m_t, t, vidx))
            lvl = torch.where(uf < t3, 3, torch.where(uf < t2, 2, torch.where(uf < t1, 1, 0)))

            soc = torch.where(rented, torch.clamp(soc - cons, min=0), soc)
            fault = torch.where(rented & (lvl > fault), lvl, fault)
            cell = torch.where(rented, dest, cell)
            idle = torch.where(rented, 0, idle + 1)

            req = demand.sum(1)
            served = rented.sum(1)
            k["requests"] += req
            k["served"] += served
            k["lost"] += req - served
            k["low_battery_steps"] += (soc < min_rent).sum(1)

    to_np = lambda x: x.reshape(K, M, *x.shape[1:]).cpu().numpy()
    return {
        "kpi": {n: to_np(v) for n, v in k.items()},
        "state": {"cell": to_np(cell), "soc": to_np(soc), "idle": to_np(idle),
                  "fault": to_np(fault), "busy": to_np(busy)},
    }
