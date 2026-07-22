from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.export import (
    DEFAULT_EXPORT_COLUMNS,
    build_export_dataframe,
    normalize_csv_filename,
    slice_results_interval,
)


def make_results() -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 00:00:00", periods=1440, freq="1min")
    df = pd.DataFrame(
        {
            "G": np.arange(1440, dtype=float),
            "G_eff": np.arange(1440, dtype=float),
            "Tamb": np.full(1440, 25.0),
            "Tc": np.full(1440, 35.0),
            "Vmp": np.full(1440, 30.0),
            "Imp": np.full(1440, 8.0),
            "Pmp": np.full(1440, 240.0),
            "Voc": np.full(1440, 38.0),
            "Isc": np.full(1440, 8.5),
            "FF": np.full(1440, 0.75),
            "eta": np.full(1440, 0.20),
            "P_array": np.arange(1440, dtype=float),
            "P_disp": np.arange(1440, dtype=float) * 2,
            "P_lineal_ref": np.arange(1440, dtype=float),
        },
        index=idx,
    )
    df.attrs["n_modules"] = 4
    return df


def test_120_minute_window_has_exactly_120_rows():
    results = make_results()
    window, start, end = slice_results_interval(results, "2026-07-22", "06:00", 120)

    assert start == pd.Timestamp("2026-07-22 06:00:00")
    assert end == pd.Timestamp("2026-07-22 08:00:00")
    assert len(window) == 120
    assert window.index[0] == pd.Timestamp("2026-07-22 06:00:00")
    assert window.index[-1] == pd.Timestamp("2026-07-22 07:59:00")
    assert window.attrs["n_modules"] == 4


def test_default_csv_has_timestamp_and_generated_power():
    results = make_results()
    window, _, _ = slice_results_interval(results, "2026-07-22", "06:00", 120)
    exported = build_export_dataframe(window, DEFAULT_EXPORT_COLUMNS, 1 / 60)

    assert exported.columns.tolist() == ["timestamp", "potencia_gerada_W"]
    assert len(exported) == 120
    assert exported.iloc[0]["timestamp"] == "2026-07-22 06:00:00"
    assert exported.iloc[0]["potencia_gerada_W"] == 360.0


def test_energy_step_uses_timestep():
    results = make_results()
    window, _, _ = slice_results_interval(results, "2026-07-22", "01:00", 1)
    exported = build_export_dataframe(window, ["Energia gerada no passo [Wh]"], 1 / 60)
    assert exported.iloc[0]["energia_passo_Wh"] == 1.0


def test_filename_normalization():
    assert normalize_csv_filename("Modelo_solar_1") == "Modelo_solar_1.csv"
    assert normalize_csv_filename("meu arquivo.csv") == "meu_arquivo.csv"
    assert normalize_csv_filename("../arquivo") == "arquivo.csv"
    assert normalize_csv_filename("") == "Modelo_solar_1.csv"
