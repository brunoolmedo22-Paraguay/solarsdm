"""
simulation/export.py
====================
Utilidades para selecionar uma janela temporal e preparar um CSV compacto para
integração com o otimizador.

A janela usa o intervalo semiaberto [início, fim). Assim, para dados de 1 minuto,
um intervalo de 120 minutos contém exatamente 120 linhas.
"""

from __future__ import annotations

from datetime import date, datetime, time
import re
from typing import Iterable

import pandas as pd


# Nome exibido na interface -> (nome estável no CSV, coluna de origem, transformação)
# A transformação recebe (série, dt_h) e devolve a série exportada.
EXPORT_COLUMN_SPECS = {
    "Timestamp": ("timestamp", None, None),
    "Potência gerada [W]": ("potencia_gerada_W", "P_array", None),
    "Energia gerada no passo [Wh]": (
        "energia_passo_Wh", "P_array", lambda s, dt_h: s.astype(float) * dt_h,
    ),
    "Irradiância GHI [W/m²]": ("ghi_W_m2", "G", None),
    "Irradiância efetiva [W/m²]": ("irradiancia_efetiva_W_m2", "G_eff", None),
    "Temperatura ambiente [°C]": ("temperatura_ambiente_C", "Tamb", None),
    "Temperatura da célula [°C]": ("temperatura_celula_C", "Tc", None),
    "Eficiência [%]": ("eficiencia_pct", "eta", lambda s, dt_h: s.astype(float) * 100.0),
    "Tensão no MPP do módulo [V]": ("vmp_modulo_V", "Vmp", None),
    "Corrente no MPP do módulo [A]": ("imp_modulo_A", "Imp", None),
    "Potência no MPP do módulo [W]": ("pmp_modulo_W", "Pmp", None),
    "Tensão de circuito aberto [V]": ("voc_modulo_V", "Voc", None),
    "Corrente de curto-circuito [A]": ("isc_modulo_A", "Isc", None),
    "Fator de forma [-]": ("fator_forma", "FF", None),
    "Potência solar incidente [W]": ("potencia_solar_incidente_W", "P_disp", None),
    "Referência linear de potência [W]": ("potencia_referencia_linear_W", "P_lineal_ref", None),
    "Dado original": ("dado_original", "is_original", None),
    "Dado preenchido": ("dado_preenchido", "is_filled", None),
    "Tipo de preenchimento": ("tipo_preenchimento", "fill_type", None),
    "Temperatura preenchida": ("temperatura_preenchida", "Tamb_filled", None),
}

DEFAULT_EXPORT_COLUMNS = ["Timestamp", "Potência gerada [W]"]


def _copy_with_attrs(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Mantém os metadados do array após filtros/cópias."""
    target.attrs = dict(source.attrs)
    return target


def available_export_columns(results: pd.DataFrame) -> list[str]:
    """Lista apenas opções cujas colunas de origem existem no resultado."""
    available: list[str] = []
    for label, (_, source_col, _) in EXPORT_COLUMN_SPECS.items():
        if source_col is None or source_col in results.columns:
            available.append(label)
    return available


def slice_results_interval(
    results: pd.DataFrame,
    selected_day: date | str | pd.Timestamp,
    start_value: time | datetime | str | pd.Timestamp,
    duration_minutes: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Seleciona [início, início + duração) dentro do dia escolhido.

    Retorna (recorte, início, fim_exclusivo). O recorte preserva ``attrs``.
    """
    if results is None or results.empty:
        raise ValueError("Não há resultados disponíveis para exportação.")
    if not isinstance(results.index, pd.DatetimeIndex):
        raise TypeError("Os resultados devem usar um DatetimeIndex.")
    if int(duration_minutes) < 1:
        raise ValueError("A duração deve ser de pelo menos 1 minuto.")

    day = pd.Timestamp(selected_day).normalize()

    if isinstance(start_value, pd.Timestamp):
        start = start_value
    elif isinstance(start_value, datetime):
        start = pd.Timestamp(start_value)
    elif isinstance(start_value, time):
        start = day + pd.Timedelta(
            hours=start_value.hour,
            minutes=start_value.minute,
            seconds=start_value.second,
        )
    else:
        parsed = pd.Timestamp(start_value)
        # Entradas contendo apenas horário são associadas ao dia selecionado.
        if parsed.date() == date(1900, 1, 1) or not any(ch in str(start_value) for ch in "-/"):
            start = day + pd.Timedelta(
                hours=parsed.hour, minutes=parsed.minute, seconds=parsed.second,
            )
        else:
            start = parsed

    if start.normalize() != day:
        start = day + (start - start.normalize())

    end = start + pd.Timedelta(minutes=int(duration_minutes))
    day_end = day + pd.Timedelta(days=1)
    effective_end = min(end, day_end)

    ordered = results.sort_index()
    mask = (ordered.index >= start) & (ordered.index < effective_end)
    window = _copy_with_attrs(results, ordered.loc[mask].copy())
    return window, start, end


def build_export_dataframe(
    results_window: pd.DataFrame,
    selected_labels: Iterable[str],
    dt_hours: float,
) -> pd.DataFrame:
    """Monta a tabela final com nomes de coluna estáveis para o otimizador."""
    labels = list(selected_labels)
    if not labels:
        raise ValueError("Selecione pelo menos uma coluna para o CSV.")

    unknown = [label for label in labels if label not in EXPORT_COLUMN_SPECS]
    if unknown:
        raise KeyError("Colunas de exportação desconhecidas: " + ", ".join(unknown))

    output = pd.DataFrame(index=results_window.index)
    for label in labels:
        csv_name, source_col, transform = EXPORT_COLUMN_SPECS[label]
        if source_col is None:
            output[csv_name] = results_window.index.strftime("%Y-%m-%d %H:%M:%S")
            continue
        if source_col not in results_window.columns:
            raise KeyError(f"A coluna necessária '{source_col}' não existe nos resultados.")
        series = results_window[source_col]
        output[csv_name] = transform(series, dt_hours) if transform else series.to_numpy()

    return output.reset_index(drop=True)


def normalize_csv_filename(value: str, default: str = "Modelo_solar_1") -> str:
    """Normaliza o nome sem permitir caminhos ou caracteres inválidos."""
    name = (value or "").strip()
    if not name:
        name = default

    if name.lower().endswith(".csv"):
        name = name[:-4]

    name = re.sub(r"[<>:\"/\\|?*]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    if not name:
        name = default
    return f"{name}.csv"
