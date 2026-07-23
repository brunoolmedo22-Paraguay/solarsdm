from pathlib import Path

import pandas as pd

from simulation.automation import (
    AUTOMATION_INPUT_FILENAMES,
    AUTOMATION_N_PARALLEL,
    AUTOMATION_N_SERIES,
    build_configuration_report,
    build_fixed_automation_module,
    build_results_zip,
    load_automation_profile,
    run_all_automation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dados_exemplo"


def test_fixed_module_configuration():
    module = build_fixed_automation_module()
    assert module.stc.model.startswith("CS7L-580MS")
    assert module.sdm.IL_ref == 18.2996
    assert module.sdm.I0_ref == 1.0494e-11
    assert module.sdm.Rs == 0.1074
    assert module.sdm.Rsh_ref == 66.2
    assert module.sdm.n == 0.9332
    assert module.sdm.n_cells == 60


def test_profiles_are_strict_120_minute_windows():
    for filename in AUTOMATION_INPUT_FILENAMES:
        profile = load_automation_profile(DATA_DIR / filename)
        assert list(profile.columns) == ["G", "Tamb"]
        assert len(profile) == 120
        diffs = profile.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=1)).all()


def test_all_cases_generate_compact_optimizer_csvs():
    module, cases = run_all_automation_cases(DATA_DIR)
    assert len(cases) == 4
    assert module.stc.p_nom * AUTOMATION_N_SERIES * AUTOMATION_N_PARALLEL == 3480.0

    expected_names = {
        "Modelo_solar_06.csv",
        "Modelo_solar_12.csv",
        "Modelo_solar_16.csv",
        "Modelo_solar_18.csv",
    }
    assert {case.output_filename for case in cases} == expected_names

    for case in cases:
        assert len(case.export_df) == 120
        assert list(case.export_df.columns) == ["timestamp", "potencia_gerada_W"]
        assert (case.export_df["potencia_gerada_W"] >= 0).all()
        assert case.csv_bytes.startswith(b"timestamp,potencia_gerada_W")

    assert len(build_results_zip(cases)) > 0


def test_configuration_report_contains_fixed_contract():
    report = build_configuration_report()
    assert "CS7L-580MS" in report
    assert "3" in report
    assert "2" in report
    assert "3480.0 Wp" in report
    assert "timestamp" in report
    assert "potencia_gerada_W" in report
