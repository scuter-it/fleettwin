"""Configuration for the FleetTwin simulator.

Every default below is synthetic. Values marked PLACEHOLDER are not
calibrated against any real operator and must be set before the results
are used for operational decisions.
"""

from dataclasses import dataclass

STEP_SECONDS = 900  # 15-minute simulation step
N_STEPS = 96        # 24-hour horizon


@dataclass(frozen=True)
class CityConfig:
    """Synthetic city on an H3 grid."""

    centre_lat: float = 45.4642  # arbitrary anchor; the demand model is synthetic
    centre_lng: float = 9.1900
    h3_resolution: int = 8
    radius_rings: int = 8         # 1 + 3k(k+1) cells: 217 cells for k = 8
    centre_rings: int = 2         # cells within this ring distance count as "centre"
    n_hotspots: int = 4
    hotspot_weight: float = 3.0
    demand_decay_rings: float = 4.0
    daily_requests: float = 500.0      # PLACEHOLDER: synthetic city-wide requests per day
    od_distance_scale_km: float = 2.5
    detour_factor: float = 1.3         # PLACEHOLDER: road distance / straight-line distance
    intra_cell_trip_km: float = 1.0
    depot_ring: int = 3                # PLACEHOLDER: depot and workshop share one cell
    event_ring: int = 5                # synthetic venue (e.g. a stadium)
    event_probability: float = 0.25    # share of scenarios that contain an event
    event_extra_requests: float = 40.0
    event_start_hour: float = 22.0
    event_duration_h: float = 1.5
    demand_levels: tuple = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
    poisson_kmax: int = 24
    forecast_lookahead_h: float = 3.0
    seed: int = 7


@dataclass(frozen=True)
class FleetConfig:
    """Fleet size, battery model and fault hazards."""

    n_vehicles: int = 200
    consumption_permille_per_km: float = 15.0  # PLACEHOLDER: state of charge (per mille) per km
    min_rentable_permille: int = 100           # PLACEHOLDER: below this a vehicle cannot be rented
    initial_soc_min: int = 150
    initial_soc_max: int = 1000
    initial_idle_hours_max: float = 72.0
    initial_onsite_faults: int = 3
    initial_workshop_faults: int = 2
    initial_incidents: int = 1
    p_fault_onsite: float = 0.010    # PLACEHOLDER: per rental (flat tyre, broken part, dirty)
    p_fault_workshop: float = 0.004  # PLACEHOLDER: per rental, non-urgent workshop job
    p_fault_incident: float = 0.002  # PLACEHOLDER: per rental, vehicle must be recovered
    seed: int = 11


@dataclass(frozen=True)
class OpsConfig:
    """Field operations: one operator, one van."""

    shifts: tuple = ((9.0, 4.0),)      # PLACEHOLDER: (start hour, duration in hours)
    van_speed_kmh: float = 18.0        # PLACEHOLDER
    swap_seconds: int = 240            # PLACEHOLDER: per vehicle, both paired batteries
    repair_onsite_seconds: int = 900   # PLACEHOLDER
    pickup_seconds: int = 420          # PLACEHOLDER: load one vehicle on the van
    drop_seconds: int = 240            # PLACEHOLDER: unload one vehicle
    reload_seconds: int = 900          # PLACEHOLDER: reload batteries at the depot
    battery_capacity: int = 15         # vehicle battery sets per van load
    workshop_repair_hours: float = 24.0
    max_jobs_per_step: int = 6


@dataclass(frozen=True)
class Policy:
    """One candidate plan: a parameterisation of the operating rules."""

    name: str
    swap_mandatory_permille: int = 300     # must swap below this
    swap_opportunistic_permille: int = 400  # may swap below this if close to a stop
    opportunistic_radius_m: int = 600
    relocate_idle_hours: float = 48.0
    relocation_target: int = 0             # 0 = centre, 1 = forecast demand deficit
    pair_radius_m: int = 1500              # second vehicle on the two-slot van
    class_penalty_m: int = 10_000_000      # large = strict priority between task classes

    def __post_init__(self):
        if self.swap_opportunistic_permille < self.swap_mandatory_permille:
            raise ValueError("swap_opportunistic_permille must be >= swap_mandatory_permille")
        if self.relocation_target not in (0, 1):
            raise ValueError("relocation_target must be 0 (centre) or 1 (demand)")
