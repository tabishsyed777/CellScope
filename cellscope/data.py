"""Synthetic cell-cycle measurements and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "cell_id", "cycle", "capacity_ah", "nominal_capacity_ah",
    "resistance_mohm", "temperature_c", "voltage_sag_mv", "charge_time_min",
]
FEATURES = [
    "cycle", "soh_pct", "capacity_change_20",
    "resistance_mohm", "temperature_c", "voltage_sag_mv", "charge_time_min",
]
ANOMALY_FEATURES = [
    "capacity_change_20", "resistance_mohm", "temperature_c",
    "voltage_sag_mv", "charge_time_min",
]


def make_demo_data(n_cells: int = 36, seed: int = 31) -> pd.DataFrame:
    """Generate transparent simulated measurements with known end-of-life labels."""
    rng = np.random.default_rng(seed)
    records = []
    anomaly_cells = set(rng.choice(np.arange(n_cells), size=max(3, n_cells // 3), replace=False))
    for number in range(n_cells):
        cell_id = f"CELL-{number + 1:03d}"
        nominal = float(rng.uniform(2.7, 3.3))
        linear_rate = float(rng.uniform(0.00019, 0.00034))
        quadratic_rate = float(rng.uniform(0.00000006, 0.00000013))
        cycles = np.arange(1, 901)
        latent_soh = 1 - linear_rate * cycles - quadratic_rate * cycles**2
        crossed = np.flatnonzero(latent_soh <= 0.80)
        end_cycle = int(crossed[0] + 1) if len(crossed) else 900
        cycles = cycles[:end_cycle]
        latent_soh = latent_soh[:end_cycle]
        capacity = nominal * latent_soh + rng.normal(0, 0.008, len(cycles))
        resistance = 17 + 0.018 * cycles + 42 * (1 - latent_soh) + rng.normal(0, 0.8, len(cycles))
        temperature = 25 + rng.normal(0, 1.3, len(cycles)) + 8 * (1 - latent_soh)
        voltage_sag = 32 + 260 * (1 - latent_soh) + rng.normal(0, 5, len(cycles))
        charge_time = 42 + 48 * (1 - latent_soh) + rng.normal(0, 2.5, len(cycles))
        anomaly = np.zeros(len(cycles), dtype=bool)

        if number in anomaly_cells and len(cycles) > 40:
            start = int(rng.integers(20, len(cycles) - 12))
            duration = int(rng.integers(4, 10))
            stop = min(len(cycles), start + duration)
            anomaly[start:stop] = True
            resistance[start:stop] += rng.uniform(28, 55)
            temperature[start:stop] += rng.uniform(8, 14)
            voltage_sag[start:stop] += rng.uniform(110, 240)
            charge_time[start:stop] += rng.uniform(5, 12)

        for i, cycle in enumerate(cycles):
            records.append({
                "cell_id": cell_id,
                "cycle": int(cycle),
                "capacity_ah": float(max(0, capacity[i])),
                "nominal_capacity_ah": nominal,
                "resistance_mohm": float(resistance[i]),
                "temperature_c": float(temperature[i]),
                "voltage_sag_mv": float(voltage_sag[i]),
                "charge_time_min": float(charge_time[i]),
                "true_remaining_cycles": int(max(0, end_cycle - cycle)),
                "anomaly_truth": bool(anomaly[i]),
            })
    return engineer_features(pd.DataFrame(records))


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort cell cycles and derive health and recent capacity trend features."""
    data = frame.copy()
    data["cell_id"] = data["cell_id"].astype("string").str.strip()
    numeric = [column for column in REQUIRED_COLUMNS if column not in {"cell_id"}]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED_COLUMNS)
    data = data[(data["nominal_capacity_ah"] > 0) & (data["cycle"] >= 1)]
    data = data[data["cell_id"].notna() & (data["cell_id"] != "")]
    data = data.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    data["soh_pct"] = 100 * data["capacity_ah"] / data["nominal_capacity_ah"]
    group = data.groupby("cell_id", sort=False)["capacity_ah"]
    trend = group.diff(20) / 20
    first_difference = group.diff()
    data["capacity_change_20"] = trend.fillna(first_difference).fillna(0.0)
    return data


def validate_upload(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return frame, [f"Missing required columns: {', '.join(missing)}"]
    clean = engineer_features(frame)
    messages = []
    counts = clean.groupby("cell_id").size()
    if clean.empty:
        messages.append("No complete numeric rows remain after cleaning.")
    elif (counts < 5).any():
        messages.append("Each cell_id needs at least five complete cycle rows.")
    if "true_remaining_cycles" in clean.columns:
        clean = clean.drop(columns="true_remaining_cycles")
    if "anomaly_truth" in clean.columns:
        clean = clean.drop(columns="anomaly_truth")
    return clean, messages
