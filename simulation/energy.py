"""
simulation/energy.py
====================
Integración energética e indicadores técnicos (KPIs) del sistema PV para
perfiles de uno o varios días.

Definiciones principales:
    E_periodo [kWh] = Σ P(t) · Δt / 1000
    H_periodo [kWh/m²] = Σ G(t) · Δt / 1000
    Yield específico [kWh/kWp] = E_periodo / P_nom[kWp]
    PR [-] = Yield específico / H_periodo
    CF [-] = E_periodo / (P_nom[kW] · duración[h])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.irradiance_model import infer_timestep_hours


def integrate_energy(power_w: pd.Series, dt_hours: float) -> np.ndarray:
    """Energía acumulada [kWh] a lo largo del período."""
    return np.cumsum(power_w.to_numpy(dtype=float) * dt_hours) / 1000.0


def compute_kpis(results: pd.DataFrame, module, dt_hours: float | None = None) -> dict:
    """Calcula KPIs consistentes para un período arbitrario de simulación."""
    stc = module.stc
    dt = infer_timestep_hours(results) if dt_hours is None else float(dt_hours)

    n_mod = results.attrs.get("n_modules", 1)
    n_series = results.attrs.get("n_series", n_mod)
    n_parallel = results.attrs.get("n_parallel", 1)
    p_nom_kw = results.attrs.get("p_nom_array_W", stc.p_nom * n_mod) / 1000.0
    area = results.attrs.get("area_array_m2", stc.area * n_mod)

    p = results["P_array"].to_numpy(dtype=float)
    g = results["G"].to_numpy(dtype=float)
    duration_h = float(len(results) * dt)
    duration_days = duration_h / 24.0 if duration_h > 0 else 0.0

    # Energía e irradiación del período completo
    e_period_kwh = float(np.sum(p) * dt / 1000.0)
    h_period = float(np.sum(g) * dt / 1000.0)
    e_day_avg = e_period_kwh / duration_days if duration_days > 0 else 0.0
    h_day_avg = h_period / duration_days if duration_days > 0 else 0.0

    # Potencias
    p_max = float(np.max(p)) if len(p) else 0.0
    i_max = int(np.argmax(p)) if len(p) else 0
    t_peak = results.index[i_max] if len(p) else None
    p_mean = float(np.mean(p)) if len(p) else 0.0
    sun = g > 0
    p_mean_sun = float(np.mean(p[sun])) if sun.any() else 0.0

    # Indicadores normalizados
    yf = e_period_kwh / p_nom_kw if p_nom_kw > 0 else 0.0
    pr = yf / h_period if h_period > 0 else 0.0
    cf = e_period_kwh / (p_nom_kw * duration_h) if p_nom_kw > 0 and duration_h > 0 else 0.0
    heq = yf

    # Eficiencia
    e_solar_kwh = h_period * area
    eta_mean = e_period_kwh / e_solar_kwh if e_solar_kwh > 0 else 0.0
    eta_inst = results["eta"].to_numpy(dtype=float)
    eta_max = float(np.max(eta_inst)) if len(eta_inst) else 0.0

    # Térmicos / eléctricos
    tc = results["Tc"].to_numpy(dtype=float)
    tc_max = float(np.max(tc)) if len(tc) else 0.0
    tc_mean_sun = float(np.mean(tc[sun])) if sun.any() else 0.0
    e_lineal_kwh = float(np.sum(results["P_lineal_ref"].to_numpy()) * dt / 1000.0)

    return {
        # Nuevas claves explícitas
        "E_period_kWh": e_period_kwh,
        "E_day_avg_kWh": e_day_avg,
        "H_period_kWh_m2": h_period,
        "H_day_avg_kWh_m2": h_day_avg,
        "duration_h": duration_h,
        "duration_days": duration_days,
        # Compatibilidad con versiones anteriores: ahora representan período
        "E_day_kWh": e_period_kwh,
        "H_sol_kWh_m2": h_period,
        "PSH_h": h_period,
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
        "area_module_m2": stc.area,
        "n_modules": n_mod,
        "n_series": n_series,
        "n_parallel": n_parallel,
        "dt_hours": dt,
    }


def energy_by_hour(results: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    """Energía producida agregada por hora cronológica [kWh]."""
    e = results["P_array"] * dt_hours / 1000.0
    by_hour = e.resample("1h").sum()
    by_hour.index.name = "timestamp"
    return by_hour.rename("E_kWh").to_frame()
