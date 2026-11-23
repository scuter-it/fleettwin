"""Synthetic city, precomputed tables and initial fleet state.

Both simulators (reference and batched) consume the same World object,
so any difference in results comes from the simulators themselves and
not from their inputs. The tables are built once, as CPU tensors.

All tables are integers: distances in metres, times in seconds, state of
charge in per mille, probabilities as thresholds on 32-bit uniforms. This
keeps the two implementations bit-identical on CPU and GPU.
"""

from dataclasses import dataclass, field
import math

import h3
import torch

from . import rng
from .config import CityConfig, FleetConfig, OpsConfig, STEP_SECONDS, N_STEPS

F64 = torch.float64

# Hourly share of daily demand (synthetic weekday profile, hours 0..23).
HOURLY_PROFILE = torch.tensor([
    0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 1.6, 2.0, 1.4, 1.0, 1.1,
    1.4, 1.3, 1.0, 1.0, 1.3, 1.8, 2.0, 1.6, 1.2, 1.0, 0.8, 0.5,
], dtype=F64)
HOURLY_PROFILE = HOURLY_PROFILE / HOURLY_PROFILE.sum()

OD_SCALE = 1 << 30  # destination uniforms live in [0, 2**30)


@dataclass
class World:
    city: CityConfig
    fleet: FleetConfig
    ops: OpsConfig
    n_steps: int
    step_seconds: int
    cells: list
    n_cells: int
    ring: torch.Tensor            # [C] ring distance from the centre cell
    in_centre: torch.Tensor       # [C] bool
    depot: int                    # depot and workshop cell (PLACEHOLDER: one cell)
    workshop: int
    event_cell: int
    dist_m: torch.Tensor          # [C*C] int64, operator road distance
    travel_s: torch.Tensor        # [C*C] int64, operator travel time
    trip_permille: torch.Tensor   # [C*C] int64, state-of-charge used by one rental
    od_flat: torch.Tensor         # [C*C] int64, offset cumulative OD thresholds
    poisson_thr: torch.Tensor     # [NV, T, C, KMAX] int64 thresholds on 32-bit uniforms
    forecast_milli: torch.Tensor  # [T, C] int64, expected requests over the lookahead x 1000
    fault_thr: tuple              # cumulative thresholds (incident, workshop, onsite)
    shift_on: torch.Tensor        # [T] bool, step starts inside a shift
    shift_start: torch.Tensor     # [T] bool, a shift starts at this step
    shift_hours: float
    init: dict = field(default_factory=dict)  # initial fleet state, int64 tensors [V]
    scenario_seed: int = 1

    @property
    def n_vehicles(self) -> int:
        return self.fleet.n_vehicles

    @property
    def n_variants(self) -> int:
        return self.poisson_thr.shape[0]

    def scenario_variants(self, scenario_ids) -> torch.Tensor:
        """Demand variant (level x event) of each scenario id."""
        m = torch.as_tensor(scenario_ids, dtype=torch.int64)
        q_count = len(self.city.demand_levels)
        q = rng.counter_hash(self.scenario_seed, rng.LEVEL, m) % q_count
        e_thr = int(self.city.event_probability * rng.U32)
        e = (rng.counter_hash(self.scenario_seed, rng.EVENT, m) < e_thr).long()
        return e * q_count + q


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0088
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = torch.deg2rad(lng2 - lng1)
    a = torch.sin(dphi / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dlmb / 2) ** 2
    return 2 * r * torch.asin(torch.sqrt(a))


def _poisson_thresholds(lam, kmax):
    """Thresholds thr[..., k] = floor(CDF(k) * 2**32); count = sum_k(u >= thr_k)."""
    pmf = torch.exp(-lam)
    cdf = pmf.clone()
    out = torch.empty(lam.shape + (kmax,), dtype=torch.int64)
    out[..., 0] = torch.floor(cdf * rng.U32)
    for k in range(1, kmax):
        pmf = pmf * lam / k
        cdf = cdf + pmf
        out[..., k] = torch.floor(torch.clamp(cdf, max=1.0) * rng.U32)
    return out


def build_world(city: CityConfig = CityConfig(), fleet: FleetConfig = FleetConfig(),
                ops: OpsConfig = OpsConfig(), n_steps: int = N_STEPS,
                scenario_seed: int = 1) -> World:
    step_s = STEP_SECONDS
    gen = torch.Generator().manual_seed(city.seed)

    # --- grid ---------------------------------------------------------------
    centre = h3.latlng_to_cell(city.centre_lat, city.centre_lng, city.h3_resolution)
    cells = sorted(h3.grid_disk(centre, city.radius_rings),
                   key=lambda c: (h3.grid_distance(centre, c), c))
    C = len(cells)
    index = {c: i for i, c in enumerate(cells)}
    ring = torch.tensor([h3.grid_distance(centre, c) for c in cells], dtype=torch.int64)
    in_centre = ring <= city.centre_rings
    latlng = torch.tensor([h3.cell_to_latlng(c) for c in cells], dtype=F64)

    straight_km = _haversine_km(latlng[:, None, 0], latlng[:, None, 1],
                                latlng[None, :, 0], latlng[None, :, 1])
    road_km = straight_km * city.detour_factor
    dist_m = torch.round(road_km * 1000).long()
    speed_ms = ops.van_speed_kmh / 3.6
    travel_s = torch.round(dist_m.double() / speed_ms).long()

    trip_km = road_km.clone()
    trip_km.fill_diagonal_(city.intra_cell_trip_km)
    trip_permille = torch.ceil(trip_km * fleet.consumption_permille_per_km).long()

    # --- special cells ------------------------------------------------------
    depot = int(torch.where(ring == min(city.depot_ring, city.radius_rings))[0][0])
    workshop = depot
    event_candidates = torch.where(ring == min(city.event_ring, city.radius_rings))[0]
    event_cell = int(event_candidates[torch.randint(len(event_candidates), (1,), generator=gen)])

    # --- demand attractiveness ----------------------------------------------
    attract = torch.exp(-ring.double() / city.demand_decay_rings)
    hot_pool = torch.where((ring >= 2) & (ring <= city.radius_rings - 1))[0]
    hotspots = hot_pool[torch.randperm(len(hot_pool), generator=gen)[:city.n_hotspots]]
    for hcell in hotspots.tolist():
        attract[hcell] += city.hotspot_weight * attract.max()
        for nb in h3.grid_ring(cells[hcell], 1):
            if nb in index:
                attract[index[nb]] += 0.5 * city.hotspot_weight * attract.max() / 6
    attract = attract / attract.sum()

    steps_per_hour = 3600 // step_s
    hours = (torch.arange(n_steps) // steps_per_hour) % 24
    lam_base = (city.daily_requests * HOURLY_PROFILE[hours] / steps_per_hour)[:, None] * attract[None, :]

    # --- demand variants: level x event ---------------------------------------
    levels = torch.tensor(city.demand_levels, dtype=F64)
    event_steps = torch.zeros(n_steps, dtype=torch.bool)
    e0 = int(city.event_start_hour * steps_per_hour)
    e1 = int((city.event_start_hour + city.event_duration_h) * steps_per_hour)
    event_steps[e0:min(e1, n_steps)] = True
    n_event_steps = max(int(event_steps.sum()), 1)
    event_lam = torch.zeros((n_steps, C), dtype=F64)
    event_lam[event_steps, event_cell] = city.event_extra_requests / n_event_steps
    lam_var = torch.cat([
        levels[:, None, None] * lam_base[None],                   # no event
        levels[:, None, None] * lam_base[None] + event_lam[None],  # event
    ])
    poisson_thr = _poisson_thresholds(lam_var, city.poisson_kmax)

    # Planner forecast: base level, no event knowledge.
    look = int(city.forecast_lookahead_h * steps_per_hour)
    idx = (torch.arange(n_steps)[:, None] + torch.arange(look)[None, :]) % n_steps
    forecast_milli = torch.round(1000 * lam_base[idx].sum(dim=1)).long()

    # --- origin-destination ---------------------------------------------------
    w = attract[None, :] * torch.exp(-trip_km / city.od_distance_scale_km)
    p = w / w.sum(dim=1, keepdim=True)
    cum = torch.floor(torch.cumsum(p, dim=1) * OD_SCALE).long()
    cum[:, -1] = OD_SCALE
    od_flat = (torch.arange(C)[:, None] * OD_SCALE + cum).reshape(-1)

    # --- faults ---------------------------------------------------------------
    t3 = int(fleet.p_fault_incident * rng.U32)
    t2 = t3 + int(fleet.p_fault_workshop * rng.U32)
    t1 = t2 + int(fleet.p_fault_onsite * rng.U32)

    # --- shifts -----------------------------------------------------------------
    shift_on = torch.zeros(n_steps, dtype=torch.bool)
    shift_start = torch.zeros(n_steps, dtype=torch.bool)
    shift_hours = 0.0
    for start_h, dur_h in ops.shifts:
        s0 = int(start_h * steps_per_hour)
        s1 = int((start_h + dur_h) * steps_per_hour)
        if s0 < n_steps:
            shift_start[s0] = True
            shift_on[s0:min(s1, n_steps)] = True
            shift_hours += (min(s1, n_steps) - s0) / steps_per_hour

    world = World(
        city=city, fleet=fleet, ops=ops, n_steps=n_steps, step_seconds=step_s,
        cells=cells, n_cells=C, ring=ring, in_centre=in_centre,
        depot=depot, workshop=workshop, event_cell=event_cell,
        dist_m=dist_m.reshape(-1), travel_s=travel_s.reshape(-1),
        trip_permille=trip_permille.reshape(-1), od_flat=od_flat,
        poisson_thr=poisson_thr, forecast_milli=forecast_milli,
        fault_thr=(t3, t2, t1), shift_on=shift_on, shift_start=shift_start,
        shift_hours=shift_hours, scenario_seed=scenario_seed,
    )
    world.init = initial_fleet(world, attract)
    return world


def initial_fleet(world: World, attract: torch.Tensor) -> dict:
    """Known fleet state at planning time (same for every plan and scenario)."""
    f = world.fleet
    gen = torch.Generator().manual_seed(f.seed)
    V, C = f.n_vehicles, world.n_cells
    placement = 0.5 * attract + 0.5 / C
    cell = torch.multinomial(placement, V, replacement=True, generator=gen)
    soc = torch.randint(f.initial_soc_min, f.initial_soc_max + 1, (V,), generator=gen)
    max_idle = int(f.initial_idle_hours_max * 3600 / world.step_seconds)
    idle = torch.randint(0, max_idle + 1, (V,), generator=gen)
    fault = torch.zeros(V, dtype=torch.int64)
    n1, n2, n3 = f.initial_onsite_faults, f.initial_workshop_faults, f.initial_incidents
    chosen = torch.randperm(V, generator=gen)[:n1 + n2 + n3]
    fault[chosen[:n1]] = 1
    fault[chosen[n1:n1 + n2]] = 2
    fault[chosen[n1 + n2:n1 + n2 + n3]] = 3
    busy = torch.zeros(V, dtype=torch.int64)
    return {"cell": cell, "soc": soc, "idle": idle, "fault": fault, "busy": busy}
