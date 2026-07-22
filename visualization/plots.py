"""
visualization/plots.py
======================
Toda la capa gráfica (Plotly). Ningún cálculo físico ocurre aquí:
estas funciones sólo reciben datos ya simulados y los dibujan.

Estilo: tema claro, paleta sobria definida en config/settings.py (UI["colors"]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import UI, G_REF

C = UI["colors"]
TPL = UI["template"]


def _base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 380):
    fig.update_layout(
        template=TPL,
        title=dict(text=title, x=0.01, font=dict(size=15)),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        height=height,
        margin=dict(l=60, r=30, t=50, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


# ===========================================================================
# 1) Irradiancia
# ===========================================================================
def plot_irradiance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["G"], name="G(t)", mode="lines",
        line=dict(color=C["irradiance"], width=1.6),
        fill="tozeroy", fillcolor="rgba(242,166,90,0.15)",
    ))
    return _base_layout(fig, "Irradiancia en el plano del módulo", "Fecha y hora", "G [W/m²]")


# ===========================================================================
# 2) Temperaturas
# ===========================================================================
def plot_temperatures(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Tamb"], name="T ambiente",
        line=dict(color=C["t_amb"], width=1.8),
    ))
    if "Tc" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Tc"], name="T célula (NOCT)",
            line=dict(color=C["t_cell"], width=1.8),
        ))
    return _base_layout(fig, "Temperatura ambiente y de célula", "Fecha y hora", "T [°C]")


# ===========================================================================
# 3) y 4) Curvas I-V y P-V
# ===========================================================================
def plot_iv_pv(curves: list[dict], title_suffix: str = "") -> go.Figure:
    """
    `curves`: lista de dicts {label, V, I, P, mpp:{Vmp,Imp,Pmp}, color}
    Dibuja I-V (eje izq.) y P-V (eje der.) superpuestas.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    palette = [C["power"], C["irradiance"], C["t_amb"], C["efficiency"], C["t_cell"], C["neutral"]]

    for k, c in enumerate(curves):
        col = c.get("color") or palette[k % len(palette)]
        fig.add_trace(go.Scatter(
            x=c["V"], y=c["I"], name=f"I-V · {c['label']}", mode="lines",
            line=dict(color=col, width=2.0),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=c["V"], y=c["P"], name=f"P-V · {c['label']}", mode="lines",
            line=dict(color=col, width=1.5, dash="dot"), opacity=0.75,
        ), secondary_y=True)

        mpp = c.get("mpp")
        if mpp and mpp["Pmp"] > 0:
            fig.add_trace(go.Scatter(
                x=[mpp["Vmp"]], y=[mpp["Imp"]], mode="markers",
                marker=dict(color=col, size=10, symbol="circle",
                            line=dict(color="white", width=1.5)),
                name=f"MPP · {c['label']}",
                hovertemplate=(f"<b>MPP {c['label']}</b><br>"
                               f"Vmp = {mpp['Vmp']:.2f} V<br>"
                               f"Imp = {mpp['Imp']:.2f} A<br>"
                               f"Pmp = {mpp['Pmp']:.1f} W<extra></extra>"),
                showlegend=False,
            ), secondary_y=False)

    fig.update_layout(
        template=TPL,
        title=dict(text=f"Curvas I-V y P-V {title_suffix}", x=0.01, font=dict(size=15)),
        height=460, margin=dict(l=60, r=60, t=50, b=45),
        legend=dict(orientation="v", y=0.98, x=0.01, bgcolor="rgba(255,255,255,0.75)"),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Tensión V [V]", gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(title_text="Corriente I [A]", secondary_y=False,
                     gridcolor=C["grid"], rangemode="tozero")
    fig.update_yaxes(title_text="Potencia P [W]", secondary_y=True,
                     showgrid=False, rangemode="tozero")
    return fig


# ===========================================================================
# 5) y 6) Potencia instantánea vs. potencia disponible
# ===========================================================================
def plot_power(df: pd.DataFrame, show_linear_ref: bool = False) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["P_disp"], name="P disponible (G·A)",
        line=dict(color=C["neutral"], width=1.2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(107,114,128,0.08)",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["P_array"], name="P salida (MPP · SDM)",
        line=dict(color=C["power"], width=2.0),
        fill="tozeroy", fillcolor="rgba(46,125,107,0.15)",
    ), secondary_y=False)

    if show_linear_ref:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["P_lineal_ref"],
            name="Referencia lineal Pnom·G/1000 (NO usada)",
            line=dict(color=C["t_cell"], width=1.2, dash="dash"),
        ), secondary_y=False)

    fig.update_layout(
        template=TPL,
        title=dict(text="Potencia instantánea y potencia solar disponible",
                   x=0.01, font=dict(size=15)),
        height=400, margin=dict(l=60, r=60, t=50, b=45), hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(title_text="Fecha y hora", gridcolor=C["grid"])
    fig.update_yaxes(title_text="P salida [W]", secondary_y=False,
                     gridcolor=C["grid"], rangemode="tozero")
    fig.update_yaxes(title_text="P solar incidente [W]", secondary_y=True,
                     showgrid=False, rangemode="tozero")
    return fig


# ===========================================================================
# 7) Eficiencia
# ===========================================================================
def plot_efficiency(df: pd.DataFrame, eta_stc: float) -> go.Figure:
    mask = df["G"] > 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[mask], y=df["eta"][mask] * 100.0, name="η(t)",
        line=dict(color=C["efficiency"], width=1.8),
    ))
    fig.add_hline(y=eta_stc * 100.0, line=dict(color=C["neutral"], dash="dash", width=1.2),
                  annotation_text=f"η STC = {eta_stc*100:.2f} %",
                  annotation_position="bottom right")
    fig = _base_layout(fig, "Eficiencia instantánea de conversión", "Fecha y hora", "η [%]")
    fig.update_yaxes(rangemode="tozero")
    return fig


# ===========================================================================
# 8) Energía acumulada + energía horaria
# ===========================================================================
def plot_energy(df: pd.DataFrame, dt_hours: float) -> go.Figure:
    """Energía por hora cronológica y energía acumulada para uno o varios días."""
    energy_step = df["P_array"].astype(float) * float(dt_hours) / 1000.0
    e_hour = energy_step.resample("1h").sum()
    e_cum = energy_step.cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=e_hour.index, y=e_hour.values,
        name="Energía por hora", marker_color=C["power_alt"], opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=e_cum.values,
        name="Energía acumulada", mode="lines",
        line=dict(color=C["energy"], width=2.2),
    ), secondary_y=True)

    fig.update_layout(
        template=TPL,
        title=dict(text="Energía producida: horaria y acumulada", x=0.01, font=dict(size=15)),
        height=400, margin=dict(l=60, r=60, t=50, b=45), hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(title_text="Fecha y hora", gridcolor=C["grid"])
    fig.update_yaxes(title_text="E por hora [kWh]", secondary_y=False, gridcolor=C["grid"])
    fig.update_yaxes(title_text="E acumulada [kWh]", secondary_y=True, showgrid=False)
    return fig


# ===========================================================================
# Cobertura temporal del CSV
# ===========================================================================
def plot_profile_coverage(df: pd.DataFrame) -> go.Figure:
    """
    Mapa fecha × hora que distingue datos originales, noche completada y
    lagunas diurnas completadas. La figura hace explícito qué valores no
    proceden directamente del CSV.
    """
    if "fill_type" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Este perfil no contiene información de cobertura.",
                           x=0.5, y=0.5, showarrow=False)
        return _base_layout(fig, "Cobertura temporal del perfil", "Hora", "Fecha", height=260)

    status_map = {
        "original": 0,
        "sintético": 0,
        "preenchido_noite": 1,
        "preenchido_lacuna_diurna": 2,
    }
    tmp = pd.DataFrame(index=df.index)
    tmp["date"] = tmp.index.strftime("%Y-%m-%d")
    tmp["minute"] = tmp.index.hour * 60 + tmp.index.minute
    tmp["status"] = df["fill_type"].map(status_map).fillna(0).astype(int).to_numpy()
    pivot = tmp.pivot_table(index="date", columns="minute", values="status", aggfunc="max")

    colorscale = [
        [0.000, "#2E7D6B"], [0.333, "#2E7D6B"],
        [0.334, "#E6E9EE"], [0.666, "#E6E9EE"],
        [0.667, "#E05C5C"], [1.000, "#E05C5C"],
    ]
    x_hours = pivot.columns.to_numpy(dtype=float) / 60.0
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=x_hours, y=pivot.index.tolist(),
        zmin=0, zmax=2, colorscale=colorscale,
        colorbar=dict(
            title="Origem", tickvals=[0, 1, 2],
            ticktext=["CSV", "Noite preenchida", "Lacuna diurna"],
            thickness=14,
        ),
        hovertemplate="Data: %{y}<br>Hora: %{x:.2f} h<br>Código: %{z}<extra></extra>",
    ))
    fig.update_layout(
        template=TPL,
        title=dict(text="Cobertura temporal e intervalos preenchidos", x=0.01, font=dict(size=15)),
        height=max(300, min(720, 180 + 4 * len(pivot))),
        margin=dict(l=80, r=150, t=50, b=45),
    )
    fig.update_xaxes(title_text="Hora do dia", range=[0, 24], dtick=2, gridcolor=C["grid"])
    fig.update_yaxes(title_text="Data", autorange="reversed", gridcolor=C["grid"])
    return fig


# ===========================================================================
# Extras de análisis
# ===========================================================================
def plot_mpp_trajectory(df: pd.DataFrame) -> go.Figure:
    """Trayectoria del MPP (Vmp, Imp) a lo largo del día, coloreada por Tc."""
    mask = df["Pmp"] > 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Vmp"][mask], y=df["Imp"][mask], mode="markers",
        marker=dict(size=5, color=df["Tc"][mask], colorscale="RdYlBu_r",
                    showscale=True, colorbar=dict(title="Tc [°C]")),
        name="MPP",
        customdata=np.stack([df["G"][mask], df["Pmp"][mask]], axis=-1),
        hovertemplate=("Vmp = %{x:.2f} V<br>Imp = %{y:.2f} A<br>"
                       "G = %{customdata[0]:.0f} W/m²<br>"
                       "Pmp = %{customdata[1]:.1f} W<extra></extra>"),
    ))
    return _base_layout(fig, "Trayectoria del MPP durante el período",
                        "Vmp [V]", "Imp [A]", height=420)


def plot_iv_family(families: dict, xlabel: str, title: str) -> go.Figure:
    """Familia de curvas I-V parametrizadas (por G o por Tc)."""
    fig = go.Figure()
    n = len(families)
    for k, (label, c) in enumerate(families.items()):
        shade = int(40 + 160 * k / max(n - 1, 1))
        fig.add_trace(go.Scatter(
            x=c["V"], y=c["I"], name=label, mode="lines",
            line=dict(color=f"rgb({shade+40},{200-shade//2},{160})", width=2),
        ))
        if c.get("mpp") and c["mpp"]["Pmp"] > 0:
            fig.add_trace(go.Scatter(
                x=[c["mpp"]["Vmp"]], y=[c["mpp"]["Imp"]], mode="markers",
                marker=dict(size=8, color="#222", symbol="x"),
                showlegend=False, hoverinfo="skip",
            ))
    return _base_layout(fig, title, xlabel, "Corriente I [A]", height=420)
