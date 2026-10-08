# FleetTwin v0.1 model

## Time and space

- 96 steps of 15 minutes (24 hours).
- H3 resolution 8: 217 cells within 8 rings of the centre by default.
- Distances are straight-line distances between cell centroids, multiplied by a detour factor. Travel times assume a constant van speed.

## Vehicle state

| Field | Meaning |
| --- | --- |
| `cell` | Current H3 cell index |
| `soc` | State of charge, per mille (both paired batteries together) |
| `idle` | Steps since the last rental |
| `fault` | 0 none, 1 on-site repair, 2 non-urgent workshop, 3 incident |
| `busy` | First step at which the vehicle is available again |

A vehicle can be rented when it has fault 0 or 2, a state of charge at or above the rentable minimum, and is not busy.

## Step order

1. **Operations**, only if a shift is on. The operator starts jobs while its clock is inside the step, up to `max_jobs_per_step`. A job's effects apply when it starts. The serviced vehicles stay busy until the step in which the job ends.
2. **Rentals.** Poisson requests per cell are matched to available vehicles in that cell, most-charged first. Each rented vehicle moves to a sampled destination within the step, loses charge in proportion to distance, and may receive a fault.

## Operator job selection

Each vehicle gets a class: 0 incident, 1 mandatory swap, 2 on-site repair, 3 relocation, or none. The next job minimises `class × class_penalty_m + distance_m`. A very large penalty gives strict priority; a small one lets a near low-priority task go before a distant high-priority one.

| Job | Duration | Effect |
| --- | --- | --- |
| Incident | Travel, pick up, optional partner pick-up, travel to workshop, drop | Vehicles repaired; busy for `workshop_repair_hours` |
| Swap | Travel, then swap time for each selected vehicle | Charge set to 1000; van batteries decrease |
| Reload | Travel to depot, reload time | Van batteries refilled |
| Repair | Travel, repair time | Fault cleared |
| Relocation | Travel, pick up one or two vehicles, travel to target, drop | Vehicles moved; idle reset |

## Simplifications to review before operational use

- A rental starts and ends within one 15-minute step.
- Customers do not walk to a neighbouring cell to find a vehicle.
- One operator and one van that both carries batteries and has two vehicle slots. In reality there may be two vans.
- The depot and the workshop share one cell.
- Jobs may run past the end of a shift.
- Faults arise only as a consequence of rentals, with fixed probabilities.
- The planner's forecast for demand-based relocation uses the base demand level and does not know about events.
- Every value marked PLACEHOLDER in `fleettwin/config.py`: consumption per km, rentable minimum, task durations, van speed, shift times, fault rates, city size and demand volume.
