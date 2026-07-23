"""
simulation/automation.py
========================
Fluxo automático e não configurável para converter previsões de GHI/Tamb em
potência fotovoltaica prevista.

A automação lê quatro CSVs versionados no repositório, aplica uma configuração
fixa do gerador e devolve um CSV compacto por janela, contendo somente:

    timestamp, potencia_gerada_W

Nenhum parâmetro desta rotina é exposto para edição na interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import zipfile

import numpy as np
import pandas as pd

from config.pv_database import get_module
from models.irradiance_model import (
    detect_profile_columns,
    infer_timestep_hours,
    read_custom_profile_table,
)
from models.pv_module import PVModule, SDMParams
from simulation.energy import compute_kpis
from simulation.export import build_export_dataframe
from simulation.mpp import simulate_timeseries


AUTOMATION_MODULE_KEY = "CS7L-580MS"
AUTOMATION_INPUT_FILENAMES = (
    "PREVISAO_SOLAR_120min_06.csv",
    "PREVISAO_SOLAR_120min_12.csv",
    "PREVISAO_SOLAR_120min_16.csv",
    "PREVISAO_SOLAR_120min_18.csv",
)
AUTOMATION_N_SERIES = 3
AUTOMATION_N_PARALLEL = 2
AUTOMATION_SOILING_LOSSES = 0.0
AUTOMATION_EXPECTED_ROWS = 120
AUTOMATION_EXPECTED_STEP_MINUTES = 1.0
AUTOMATION_EXPORT_COLUMNS = ("Timestamp", "Potência gerada [W]")

# Parâmetros ajustados fornecidos para o caso automático.
AUTOMATION_SDM = SDMParams(
    IL_ref=18.2996,
    I0_ref=1.0494e-11,
    Rs=0.1074,
    Rsh_ref=66.2,
    n=0.9332,
    n_cells=60,
    source="fixed_automation",
)


@dataclass
class AutomationCaseResult:
    """Resultado completo de uma janela automática."""

    input_filename: str
    output_filename: str
    run_hour: str
    profile: pd.DataFrame
    results: pd.DataFrame
    export_df: pd.DataFrame
    csv_bytes: bytes
    kpis: dict

    @property
    def start(self) -> pd.Timestamp:
        return self.profile.index.min()

    @property
    def end(self) -> pd.Timestamp:
        return self.profile.index.max()


def build_fixed_automation_module() -> PVModule:
    """Cria o CS7L-580MS com os parâmetros SDM fixos da automação."""
    module = get_module(AUTOMATION_MODULE_KEY)
    module.sdm = SDMParams(**AUTOMATION_SDM.to_dict())
    return module


def _find_required_columns(raw: pd.DataFrame) -> tuple[str, str, str]:
    detected = detect_profile_columns(raw.columns)
    timestamp_col = detected.get("timestamp_default")
    irradiance_col = detected.get("irradiance_default")
    temperature_col = detected.get("temperature_default")

    missing: list[str] = []
    if timestamp_col is None:
        missing.append("timestamp")
    if irradiance_col is None:
        missing.append("GHI")
    if temperature_col is None:
        missing.append("Tamb")
    if missing:
        raise ValueError(
            "O CSV automático não contém todas as colunas obrigatórias: "
            + ", ".join(missing)
        )
    return str(timestamp_col), str(irradiance_col), str(temperature_col)


def load_automation_profile(path: str | Path) -> pd.DataFrame:
    """
    Lê e valida uma previsão de 120 minutos.

    A rotina é deliberadamente estrita: não preenche dados, não cria horários e
    não gera temperatura. O CSV deve chegar pronto do bloco de previsão.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Arquivo de previsão não encontrado: {csv_path}")

    raw = read_custom_profile_table(csv_path)
    timestamp_col, irradiance_col, temperature_col = _find_required_columns(raw)

    timestamp = pd.to_datetime(raw[timestamp_col], errors="coerce")
    g = pd.to_numeric(raw[irradiance_col], errors="coerce")
    tamb = pd.to_numeric(raw[temperature_col], errors="coerce")

    invalid_timestamp = int(timestamp.isna().sum())
    invalid_g = int(g.isna().sum())
    invalid_t = int(tamb.isna().sum())
    if invalid_timestamp or invalid_g or invalid_t:
        raise ValueError(
            f"CSV inválido ({csv_path.name}): timestamp inválido={invalid_timestamp}, "
            f"GHI inválido={invalid_g}, Tamb inválida={invalid_t}."
        )

    profile = pd.DataFrame(
        {"G": g.to_numpy(dtype=float), "Tamb": tamb.to_numpy(dtype=float)},
        index=pd.DatetimeIndex(timestamp, name="timestamp"),
    ).sort_index()

    if profile.index.has_duplicates:
        duplicates = int(profile.index.duplicated().sum())
        raise ValueError(
            f"CSV inválido ({csv_path.name}): existem {duplicates} timestamps duplicados."
        )
    if len(profile) != AUTOMATION_EXPECTED_ROWS:
        raise ValueError(
            f"CSV inválido ({csv_path.name}): esperado {AUTOMATION_EXPECTED_ROWS} linhas, "
            f"recebido {len(profile)}."
        )
    if (profile["G"] < 0).any():
        raise ValueError(f"CSV inválido ({csv_path.name}): GHI não pode ser negativo.")
    if not np.isfinite(profile[["G", "Tamb"]].to_numpy()).all():
        raise ValueError(f"CSV inválido ({csv_path.name}): existem valores não finitos.")

    diffs_minutes = profile.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    if not np.allclose(
        diffs_minutes.to_numpy(dtype=float),
        AUTOMATION_EXPECTED_STEP_MINUTES,
        rtol=0.0,
        atol=1e-9,
    ):
        observed = sorted(set(np.round(diffs_minutes.to_numpy(dtype=float), 6)))
        raise ValueError(
            f"CSV inválido ({csv_path.name}): a discretização deve ser de 1 minuto. "
            f"Passos observados: {observed[:8]}."
        )

    # Metadados usados pela interface e pelos relatórios.
    profile.attrs["source_file"] = csv_path.name
    profile.attrs["run_hour"] = _extract_run_hour(csv_path.name, profile.index[0])
    return profile


def _extract_run_hour(filename: str, first_timestamp: pd.Timestamp) -> str:
    """Extrai HH do padrão *_HH.csv; usa o primeiro timestamp como fallback."""
    stem = Path(filename).stem
    token = stem.rsplit("_", 1)[-1]
    if token.isdigit() and 0 <= int(token) <= 23:
        return f"{int(token):02d}"
    # A primeira previsão começa no minuto seguinte à execução.
    execution_time = first_timestamp - pd.Timedelta(minutes=1)
    return execution_time.strftime("%H")


def run_automation_case(path: str | Path, module: PVModule | None = None) -> AutomationCaseResult:
    """Executa uma das janelas fixas e prepara seu CSV compacto."""
    profile = load_automation_profile(path)
    fixed_module = build_fixed_automation_module() if module is None else module

    results = simulate_timeseries(
        fixed_module,
        profile,
        noct=fixed_module.stc.noct,
        n_series=AUTOMATION_N_SERIES,
        n_parallel=AUTOMATION_N_PARALLEL,
        soiling_losses=AUTOMATION_SOILING_LOSSES,
    )
    dt_hours = infer_timestep_hours(profile)
    export_df = build_export_dataframe(
        results,
        selected_labels=AUTOMATION_EXPORT_COLUMNS,
        dt_hours=dt_hours,
    )
    run_hour = str(profile.attrs["run_hour"])
    output_filename = f"Modelo_solar_{run_hour}.csv"
    csv_bytes = export_df.to_csv(
        index=False,
        float_format="%.6f",
    ).encode("utf-8")
    kpis = compute_kpis(results, fixed_module, dt_hours)

    return AutomationCaseResult(
        input_filename=Path(path).name,
        output_filename=output_filename,
        run_hour=run_hour,
        profile=profile,
        results=results,
        export_df=export_df,
        csv_bytes=csv_bytes,
        kpis=kpis,
    )


def run_all_automation_cases(
    data_dir: str | Path,
    filenames: Iterable[str] = AUTOMATION_INPUT_FILENAMES,
) -> tuple[PVModule, list[AutomationCaseResult]]:
    """Executa todos os CSVs fixos da pasta Dados_exemplo."""
    base = Path(data_dir)
    module = build_fixed_automation_module()
    cases = [run_automation_case(base / filename, module=module) for filename in filenames]
    return module, cases


def build_results_zip(cases: Iterable[AutomationCaseResult]) -> bytes:
    """Compacta os CSVs de saída mantendo os nomes Modelo_solar_HH.csv."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case in cases:
            archive.writestr(case.output_filename, case.csv_bytes)
    return buffer.getvalue()


def build_configuration_report(module: PVModule | None = None) -> str:
    """Gera relatório Markdown da configuração imutável usada na automação."""
    fixed_module = build_fixed_automation_module() if module is None else module
    stc = fixed_module.stc
    sdm = fixed_module.sdm
    n_modules = AUTOMATION_N_SERIES * AUTOMATION_N_PARALLEL

    vmp_array = stc.v_mp * AUTOMATION_N_SERIES
    voc_array = stc.v_oc * AUTOMATION_N_SERIES
    imp_array = stc.i_mp * AUTOMATION_N_PARALLEL
    isc_array = stc.i_sc * AUTOMATION_N_PARALLEL
    p_nom_array = stc.p_nom * n_modules
    area_array = stc.area * n_modules

    return f"""# Relatório de configuração — Automação solar

## Objetivo
Converter automaticamente cada previsão de 120 minutos de `GHI` e `Tamb` em
uma previsão de potência fotovoltaica minuto a minuto.

## Contrato de entrada
- Pasta do repositório: `Dados_exemplo/`
- Arquivos: `{', '.join(AUTOMATION_INPUT_FILENAMES)}`
- Colunas obrigatórias: `timestamp`, `GHI`, `Tamb`
- Horizonte: {AUTOMATION_EXPECTED_ROWS} minutos
- Discretização: {AUTOMATION_EXPECTED_STEP_MINUTES:.0f} minuto
- O arquivo contém somente dados futuros; não são usados dados históricos nesta etapa.

## Módulo fotovoltaico fixo
- Fabricante: {stc.manufacturer}
- Modelo: {stc.model}
- Tecnologia: {stc.technology}
- Potência nominal: {stc.p_nom:.1f} W
- Voc: {stc.v_oc:.3f} V
- Isc: {stc.i_sc:.3f} A
- Vmp: {stc.v_mp:.3f} V
- Imp: {stc.i_mp:.3f} A
- Número elétrico de células em série: {stc.n_cells}
- Coeficiente de Isc: {stc.alpha_isc_pct:.4f} %/°C
- Coeficiente de Voc: {stc.beta_voc_pct:.4f} %/°C
- Coeficiente de Pmax: {stc.gamma_pmax_pct:.4f} %/°C
- NOCT/NMOT: {stc.noct:.1f} °C
- Dimensões: {stc.length:.3f} m × {stc.width:.3f} m
- Área por módulo: {stc.area:.4f} m²
- Eficiência STC: {stc.efficiency_stc*100:.3f} %
- Observação de catálogo: {stc.notes}

## Parâmetros fixos do Single Diode Model
- Iph = IL: {sdm.IL_ref:.4f} A
- I0: {sdm.I0_ref:.4e} A
- log10(I0): {np.log10(sdm.I0_ref):.6f}
- Rs: {sdm.Rs:.4f} Ω
- Rsh: {sdm.Rsh_ref:.4f} Ω
- n: {sdm.n:.4f}
- Ns: {sdm.n_cells}
- a_ref = n·Ns·k·T/q: {sdm.a_ref:.4f} V
- Vt da célula em 25 °C: {sdm.vt_ref*1000:.3f} mV
- Origem: configuração fixa da automação

## Arranjo fotovoltaico fixo
- Módulos em série por string: {AUTOMATION_N_SERIES}
- Strings em paralelo: {AUTOMATION_N_PARALLEL}
- Total de módulos: {n_modules}
- Potência instalada: {p_nom_array:.1f} Wp ({p_nom_array/1000:.3f} kWp)
- Vmp nominal aproximado do arranjo: {vmp_array:.3f} V
- Imp nominal aproximada do arranjo: {imp_array:.3f} A
- Voc nominal aproximada do arranjo: {voc_array:.3f} V
- Isc nominal aproximada do arranjo: {isc_array:.3f} A
- Área total: {area_array:.4f} m²
- Perdas por sujeira configuradas: {AUTOMATION_SOILING_LOSSES*100:.1f} %
- Hipótese operacional: MPPT ideal; o arranjo opera no MPP calculado pelo SDM.
- Inversor/conversor não modelado nesta rotina.

## Saída automática
Para cada arquivo de entrada é gerado um arquivo `Modelo_solar_HH.csv` com
exatamente duas colunas:

1. `timestamp`
2. `potencia_gerada_W`
"""
