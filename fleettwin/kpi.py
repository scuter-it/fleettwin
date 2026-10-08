"""Reduction of per-(plan, scenario) KPIs to per-plan statistics.

v0.1 runs this step on the host with NumPy. Moving the reductions over
the scenario axis (means, quantiles) to the device is part of the
hackathon work.
"""

import numpy as np

METRICS = (
    ("moved_per_shift_h", "Vehicles moved / shift h"),
    ("interventions_per_shift_h", "Interventions / shift h"),
    ("service_rate", "Requests served (share)"),
    ("lost", "Requests lost"),
    ("op_km", "Operator km"),
    ("low_battery_vh", "Low-battery vehicle-hours"),
)


def derived(kpi: dict, world) -> dict:
    """Operational metrics, arrays of shape [K, M]."""
    sh = max(world.shift_hours, 1e-9)
    moved = kpi["relocated"] + kpi["recovered"]
    interventions = moved + kpi["swaps"] + kpi["repairs"]
    return {
        "moved_per_shift_h": moved / sh,
        "interventions_per_shift_h": interventions / sh,
        "service_rate": kpi["served"] / np.maximum(kpi["requests"], 1),
        "lost": kpi["lost"].astype(float),
        "op_km": kpi["op_m"] / 1000.0,
        "low_battery_vh": kpi["low_battery_steps"] * world.step_seconds / 3600.0,
    }


def summarise(kpi: dict, world, names: list) -> list:
    """Mean, 10th and 90th percentile over scenarios, one row per plan."""
    d = derived(kpi, world)
    rows = []
    for i, name in enumerate(names):
        row = {"plan": name}
        for key, _ in METRICS:
            x = d[key][i]
            row[key] = (float(x.mean()), float(np.quantile(x, 0.1)), float(np.quantile(x, 0.9)))
        rows.append(row)
    return rows


def format_table(rows: list) -> str:
    head = ["plan"] + [label for _, label in METRICS]
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        cells = [r["plan"]]
        for key, _ in METRICS:
            mean, lo, hi = r[key]
            fmt = "{:.3f}" if key == "service_rate" else "{:.1f}"
            cells.append(f"{fmt.format(mean)} [{fmt.format(lo)}-{fmt.format(hi)}]")
        lines.append(" | ".join(cells))
    return "\n".join(lines)
