"""NumPy reference simulator: one plan, one scenario, readable control flow.

This is the ground truth. The batched PyTorch simulator must reproduce
its KPIs and final state exactly for the same (plan, scenario) pair.

Vehicle state (int64, per vehicle):
  cell   H3 cell index
  soc    state of charge, per mille (0..1000)
  idle   steps since the last rental
  fault  0 none, 1 on-site repair, 2 non-urgent workshop, 3 incident
  busy   first step at which the vehicle is free again

Each step: (1) field operations if a shift is on, (2) rentals.
"""

import numpy as np

from . import rng
from .world import OD_SCALE, World

INF = np.int64(1) << np.int64(62)
KPI_NAMES = ("requests", "served", "lost", "swaps", "repairs", "relocated", "recovered",
             "reloads", "jobs", "op_m", "low_battery_steps")


def _key_unique(values, V, vidx):
    return values * V + vidx


def simulate_reference(world: World, plan: dict, m: int) -> dict:
    """Simulate plan (dict of scalar parameters) under scenario id m."""
    C, V, T, S = world.n_cells, world.n_vehicles, world.n_steps, world.step_seconds
    ops, fl = world.ops, world.fleet
    dist, trav = world.dist_m, world.travel_s
    in_centre = world.in_centre
    variant = int(world.scenario_variants([m])[0])
    workshop_steps = int(round(ops.workshop_repair_hours * 3600 / S))
    min_rent = fl.min_rentable_permille
    t3, t2, t1 = world.fault_thr

    cell = world.init["cell"].copy()
    soc = world.init["soc"].copy()
    idle = world.init["idle"].copy()
    fault = world.init["fault"].copy()
    busy = world.init["busy"].copy()
    vidx = np.arange(V, dtype=np.int64)
    cidx = np.arange(C, dtype=np.int64)

    op_cell, op_free, batteries = world.depot, 0, ops.battery_capacity
    k = dict.fromkeys(KPI_NAMES, 0)

    for t in range(T):
        step_start, step_end = t * S, (t + 1) * S
        if world.shift_start[t]:
            op_cell, batteries = world.depot, ops.battery_capacity
            op_free = max(op_free, step_start)

        # ---------------- field operations ----------------
        if world.shift_on[t]:
            for _ in range(ops.max_jobs_per_step):
                if op_free >= step_end:
                    break
                start = max(op_free, step_start)
                free = busy <= t
                d_op = dist[op_cell * C + cell]
                below_mand = soc < plan["mand"]
                c0 = free & (fault == 3)
                c1 = free & (fault == 0) & below_mand
                c2 = free & (fault == 1)
                c3 = (free & (fault == 0) & ~below_mand & (idle >= plan["idle"])
                      & ~in_centre[cell])
                cls = np.full(V, 4, dtype=np.int64)
                cls[c3] = 3
                cls[c2] = 2
                cls[c1] = 1
                cls[c0] = 0
                key = np.where(cls < 4, _key_unique(cls * plan["pen"] + d_op, V, vidx), INF)
                p = int(np.argmin(key))
                if key[p] >= INF:
                    op_free = step_end  # nothing to do: wait for the next step
                    break
                p_cls, p_cell = int(cls[p]), int(cell[p])
                d_p = dist[p_cell * C + cell]

                if p_cls in (0, 3):
                    # Two-slot van: primary vehicle plus the nearest compatible partner.
                    if p_cls == 0:
                        elig_q = free & (fault >= 2)
                    else:
                        elig_q = c3.copy()
                    elig_q[p] = False
                    elig_q &= d_p <= plan["pair_r"]
                    key_q = np.where(elig_q, _key_unique(d_p, V, vidx), INF)
                    q = int(np.argmin(key_q))
                    has_q = bool(key_q[q] < INF)
                    if p_cls == 0:
                        drop = world.workshop
                    else:
                        avail = (((fault == 0) | (fault == 2)) & (soc >= min_rent) & free)
                        counts = np.bincount(cell[avail], minlength=C).astype(np.int64)
                        if plan["target"] == 0:
                            key_t = np.where(in_centre, counts * C + cidx, INF)
                        else:
                            deficit = world.forecast_milli[t] - 1000 * counts
                            key_t = (np.int64(10**9) - deficit) * C + cidx
                        drop = int(np.argmin(key_t))
                    last = int(cell[q]) if has_q else p_cell
                    dur = trav[op_cell * C + p_cell] + ops.pickup_seconds
                    meters = dist[op_cell * C + p_cell]
                    if has_q:
                        dur += trav[p_cell * C + last] + ops.pickup_seconds
                        meters += dist[p_cell * C + last]
                    dur += trav[last * C + drop] + ops.drop_seconds * (2 if has_q else 1)
                    meters += dist[last * C + drop]
                    end = start + int(dur)
                    end_step = -(-end // S)
                    moved = [p, q] if has_q else [p]
                    for v in moved:
                        cell[v] = drop
                        idle[v] = 0
                        if p_cls == 0:
                            fault[v] = 0
                            busy[v] = end_step + workshop_steps
                        else:
                            busy[v] = end_step
                    if p_cls == 0:
                        k["recovered"] += len(moved)
                    else:
                        k["relocated"] += len(moved)
                    op_cell = drop

                elif p_cls == 1 and batteries == 0:
                    dur = trav[op_cell * C + world.depot] + ops.reload_seconds
                    meters = dist[op_cell * C + world.depot]
                    end = start + int(dur)
                    batteries = ops.battery_capacity
                    op_cell = world.depot
                    k["reloads"] += 1

                elif p_cls == 1:
                    elig_s = free & (fault == 0) & (soc < plan["opp"]) & (d_p <= plan["opp_r"])
                    key_s = np.where(elig_s, _key_unique(d_p, V, vidx), INF)
                    key_s[p] = -1  # the primary vehicle is always served first
                    order = np.argsort(key_s, kind="stable")
                    rank = np.empty(V, dtype=np.int64)
                    rank[order] = np.arange(V)
                    sel = elig_s & (rank < batteries)
                    n = int(sel.sum())
                    dur = (trav[op_cell * C + p_cell] + trav[p_cell * C + cell[sel]].sum()
                           + ops.swap_seconds * n)
                    meters = dist[op_cell * C + p_cell] + d_p[sel].sum()
                    end = start + int(dur)
                    end_step = -(-end // S)
                    soc[sel] = 1000
                    busy[sel] = end_step
                    batteries -= n
                    op_cell = p_cell
                    k["swaps"] += n

                else:  # p_cls == 2, on-site repair
                    dur = trav[op_cell * C + p_cell] + ops.repair_onsite_seconds
                    meters = dist[op_cell * C + p_cell]
                    end = start + int(dur)
                    end_step = -(-end // S)
                    fault[p] = 0
                    busy[p] = end_step
                    op_cell = p_cell
                    k["repairs"] += 1

                op_free = end
                k["jobs"] += 1
                k["op_m"] += int(meters)

        # ---------------- rentals ----------------
        avail = ((fault == 0) | (fault == 2)) & (soc >= min_rent) & (busy <= t)
        u = rng.counter_hash(world.scenario_seed, rng.DEMAND, m, t, cidx)
        thr = world.poisson_thr[variant, t]
        demand = (u[:, None] >= thr).sum(axis=1).astype(np.int64)
        cellkey = np.where(avail, cell, C)
        key = cellkey * (1001 * V) + (1000 - soc) * V + vidx
        order = np.argsort(key, kind="stable")
        scell = cellkey[order]
        is_start = np.ones(V, dtype=bool)
        is_start[1:] = scell[1:] != scell[:-1]
        pos = np.arange(V, dtype=np.int64)
        seg_start = np.maximum.accumulate(np.where(is_start, pos, 0))
        rank = np.empty(V, dtype=np.int64)
        rank[order] = pos - seg_start
        demand_ext = np.append(demand, 0)
        rented = avail & (rank < demand_ext[cellkey])

        u_dest = rng.counter_hash(world.scenario_seed, rng.DEST, m, t, vidx) >> 2
        dest = np.searchsorted(world.od_flat, cell * OD_SCALE + u_dest, side="right") - cell * C
        cons = world.trip_permille[cell * C + dest]
        uf = rng.counter_hash(world.scenario_seed, rng.FAULT, m, t, vidx)
        lvl = np.where(uf < t3, 3, np.where(uf < t2, 2, np.where(uf < t1, 1, 0)))

        soc = np.where(rented, np.maximum(soc - cons, 0), soc)
        fault = np.where(rented & (lvl > fault), lvl, fault)
        cell = np.where(rented, dest, cell)
        idle = np.where(rented, 0, idle + 1)

        served = int(rented.sum())
        req = int(demand.sum())
        k["requests"] += req
        k["served"] += served
        k["lost"] += req - served
        k["low_battery_steps"] += int((soc < min_rent).sum())

    state = {"cell": cell, "soc": soc, "idle": idle, "fault": fault, "busy": busy}
    return {"kpi": k, "state": state}
