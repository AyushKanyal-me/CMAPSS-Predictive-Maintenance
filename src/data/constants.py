"""Shared constants for NASA C-MAPSS tabular files."""

from __future__ import annotations

SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]

C_MAPSS_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *SENSOR_COLUMNS,
]

# Documented in the project decision log and commonly dropped in C-MAPSS tutorials.
RECOMMENDED_DROPPED_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
