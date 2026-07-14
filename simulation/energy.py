"""
simulation/energy.py
====================
Integración energética e indicadores técnicos (KPIs) del sistema PV.

Definiciones utilizadas (IEC 61724 / práctica de ingeniería):

    E_day   [kWh]      = Σ P(t) * Δt / 1000                (integración rectangular)
    H_sol   [kWh/m2]   = Σ G(t) * Δt / 1000                (irradiación en el plano)
    PSH     [h]        = H_sol / 1 kW/m2                   ("horas sol pico")

    Yield específico  Y_f [kWh/kWp] = E_day / P_nom[kW]
    Performance Ratio PR  [-]       = Y_f / PSH
    Factor de capacidad  CF [-]     = E_day / (P_nom[kW] * 24 h)
    Horas equivalentes   [h]        = E_day / P_nom[kW]  ( == Y_f numéricamente)
    Eficiencia media (ponderada por energía) = E_day / (H_sol * Área)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import G_REF
from models.irradiance_model import infer_timestep_hours


def integrate_energy(power_w: pd.Series, dt_hours: float) -> np.ndarray:
    """Energía acumulada [kWh] a lo largo del día (serie acumulada)."""
    return np.cumsum(power_w.to_numpy(dtype=float) * dt_hours) / 1000.0


def compute_kpis(results: pd.DataFrame, module, dt_hours: float | None = None) -> dict:
    """
    Calcula todos los KPIs a partir del DataFrame devuelto por simulate_timeseries.

    Returns
    -------
    dict con las claves usadas por la interfaz (unidades explícitas).
    """
    stc = module.stc
    dt = infer_timestep_hours(results) if dt_hours is None else float(dt_hours)

    n_mod = results.attrs.get("n_modules", 1)
    p_nom_kw = results.attrs.get("p_nom_array_W", stc.p_nom * n_mod) / 1000.0
    area = results.attrs.get("area_array_m2", stc.area * n_mod)

    p = results["P_array"].to_numpy(dtype=float)          # [W]
    g = results["G"].to_numpy(dtype=float)                # [W/m2]

    # --- Energía ----------------------------------------------------------
    e_day_kwh = float(np.sum(p) * dt / 1000.0)            # [kWh]
    h_sol = float(np.sum(g) * dt / 1000.0)                # [kWh/m2] == PSH [h]

    # --- Potencias --------------------------------------------------------
    p_max = float(np.max(p)) if len(p) else 0.0
    i_max = int(np.argmax(p)) if len(p) else 0
    t_peak = results.index[i_max] if len(p) else None
    p_mean = float(np.mean(p)) if len(p) else 0.0
    # Potencia media en horas de sol (G > 0)
    sun = g > 0
    p_mean_sun = float(np.mean(p[sun])) if sun.any() else 0.0

    # --- Indicadores normalizados ----------------------------------------
    yf = e_day_kwh / p_nom_kw if p_nom_kw > 0 else 0.0               # [kWh/kWp]
    pr = yf / h_sol if h_sol > 0 else 0.0                            # [-]
    cf = e_day_kwh / (p_nom_kw * 24.0) if p_nom_kw > 0 else 0.0      # [-]
    heq = yf                                                          # [h]

    # --- Eficiencia -------------------------------------------------------
    e_solar_kwh = h_sol * area                    # energía solar incidente [kWh]
    eta_mean = e_day_kwh / e_solar_kwh if e_solar_kwh > 0 else 0.0
    eta_inst = results["eta"].to_numpy(dtype=float)
    eta_max = float(np.max(eta_inst)) if len(eta_inst) else 0.0

    # --- Térmicos / eléctricos --------------------------------------------
    tc = results["Tc"].to_numpy(dtype=float)
    tc_max = float(np.max(tc)) if len(tc) else 0.0
    tc_mean_sun = float(np.mean(tc[sun])) if sun.any() else 0.0

    # Pérdida térmica: energía perdida vs. operar todo el día a 25 °C
    # (referencia física, no una aproximación del modelo)
    e_lineal_kwh = float(np.sum(results["P_lineal_ref"].to_numpy()) * dt / 1000.0)

    return {
        "E_day_kWh": e_day_kwh,
        "H_sol_kWh_m2": h_sol,
        "PSH_h": h_sol,
        "specific_yield_kWh_kWp": yf,
        "PR": pr,
        "CF": cf,
        "equivalent_hours_h": heq,
        "P_max_W": p_max,
        "t_peak": t_peak,
        "P_mean_W": p_mean,
        "P_mean_sun_W": p_mean_sun,
        "eta_mean": eta_mean,
        "eta_max": eta_max,
        "eta_stc": stc.efficiency_stc,
        "Tc_max_C": tc_max,
        "Tc_mean_sun_C": tc_mean_sun,
        "E_solar_kWh": e_solar_kwh,
        "E_lineal_ref_kWh": e_lineal_kwh,
        "p_nom_kWp": p_nom_kw,
        "area_m2": area,
        "n_modules": n_mod,
        "dt_hours": dt,
    }


def energy_by_hour(results: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    """Energía producida agregada por hora [kWh] (para el gráfico de barras)."""
    e = results["P_array"] * dt_hours / 1000.0
    by_hour = e.groupby(results.index.hour).sum()
    by_hour.index.name = "hora"
    return by_hour.rename("E_kWh").to_frame()
