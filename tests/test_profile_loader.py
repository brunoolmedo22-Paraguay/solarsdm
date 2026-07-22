"""Pruebas mínimas del cargador de perfiles CSV."""

from __future__ import annotations

import io
import unittest

import pandas as pd

from models.irradiance_model import (
    detect_profile_columns,
    prepare_custom_profile,
    read_custom_profile_table,
)


class ProfileLoaderTests(unittest.TestCase):
    def setUp(self):
        self.csv = io.StringIO(
            "Timestamp,GHI_REAL,GHI_PREDITO\n"
            "2026-01-01 06:00:00,5,8\n"
            "2026-01-01 06:01:00,10,12\n"
            "2026-01-01 18:00:00,0,2\n"
        )

    def test_detects_predicted_ghi_first(self):
        raw = read_custom_profile_table(self.csv)
        detected = detect_profile_columns(raw.columns)
        self.assertEqual(detected["timestamp_default"], "Timestamp")
        self.assertEqual(detected["irradiance_default"], "GHI_PREDITO")
        self.assertIsNone(detected["temperature_default"])

    def test_completes_day_and_marks_origin(self):
        raw = read_custom_profile_table(self.csv)
        out, meta = prepare_custom_profile(
            raw,
            timestamp_col="Timestamp",
            irradiance_col="GHI_PREDITO",
            complete_days=True,
            temperature_strategy="constant_day_night",
            temp_day=30.0,
            temp_night=20.0,
        )
        self.assertEqual(len(out), 1440)
        self.assertEqual(out.index.min(), pd.Timestamp("2026-01-01 00:00:00"))
        self.assertEqual(out.index.max(), pd.Timestamp("2026-01-01 23:59:00"))
        self.assertEqual(out.loc["2026-01-01 00:00:00", "G"], 0.0)
        self.assertEqual(out.loc["2026-01-01 00:00:00", "fill_type"], "preenchido_noite")
        self.assertEqual(out.loc["2026-01-01 12:00:00", "fill_type"], "preenchido_lacuna_diurna")
        self.assertEqual(out.loc["2026-01-01 12:00:00", "Tamb"], 30.0)
        self.assertEqual(meta["original_rows"], 3)
        self.assertEqual(meta["total_rows"], 1440)

    def test_standard_temperature_curve(self):
        raw = read_custom_profile_table(self.csv)
        out, _ = prepare_custom_profile(
            raw,
            timestamp_col="Timestamp",
            irradiance_col="GHI_PREDITO",
            complete_days=True,
            temperature_strategy="standard_curve",
            season="Otoño/Primavera",
        )
        self.assertFalse(out["Tamb"].isna().any())
        self.assertGreater(out["Tamb"].max(), out["Tamb"].min())


if __name__ == "__main__":
    unittest.main()
