"""
models/irradiance_model.py
==========================
Generación de perfiles temporales de irradiancia G(t) y temperatura ambiente
Tamb(t), y carga/validación de perfiles personalizados desde CSV.

Perfiles sintéticos de irradiancia:
    * "Día soleado"  : campana solar suave (seno elevado, clear-sky simplificado)
    * "Día nublado"  : envolvente de cielo claro + tránsito de nubes (caídas
                       correlacionadas + ruido) -> alta variabilidad
    * "Día lluvioso" : irradiancia baja y difusa, con ruido

Perfil térmico:
    Tamb(t) sinusoidal con mínimo al amanecer y máximo a media tarde
    (retardo térmico respecto al mediodía solar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import PROFILES, UI

IRRADIANCE_PROFILES = ["Día soleado", "Día nublado", "Día lluvioso"]
SEASONS = list(PROFILES["seasons"].keys())


# ===========================================================================
# Eje temporal
# ===========================================================================
def time_axis(timestep_min: int = PROFILES["timestep_min"], date: str = "2025-01-01"):
    """Devuelve (índice datetime, horas decimales) para un día completo."""
    n = int(PROFILES["minutes_per_day"] / timestep_min)
    idx = pd.date_range(start=date, periods=n, freq=f"{timestep_min}min")
    hours = idx.hour + idx.minute / 60.0 + idx.second / 3600.0
    return idx, np.asarray(hours, dtype=float)


# ===========================================================================
# Irradiancia
# ===========================================================================
def _clear_sky_envelope(hours: np.ndarray, sunrise: float, sunset: float, g_peak: float):
    """Campana solar: G = Gpeak * sin(pi * (t - amanecer)/(duración del día))**1.3."""
    day_len = sunset - sunrise
    x = (hours - sunrise) / day_len
    g = np.where(
        (x > 0) & (x < 1),
        g_peak * np.sin(np.pi * np.clip(x, 0, 1)) ** 1.3,
        0.0,
    )
    return g


def generate_irradiance(
    profile: str,
    hours: np.ndarray,
    sunrise: float = PROFILES["sunrise_h"],
    sunset: float = PROFILES["sunset_h"],
    g_peak: float | None = None,
    seed: int | None = PROFILES["seed"],
) -> np.ndarray:
    """Genera G(t) [W/m2] según el perfil característico solicitado."""
    rng = np.random.default_rng(seed)

    if profile == "Día soleado":
        peak = g_peak or PROFILES["g_peak_clear"]
        g = _clear_sky_envelope(hours, sunrise, sunset, peak)
        # Rugosidad muy leve (aerosoles / medición)
        g = g * (1.0 + 0.01 * rng.standard_normal(g.size))

    elif profile == "Día nublado":
        peak = g_peak or PROFILES["g_peak_cloudy"]
        env = _clear_sky_envelope(hours, sunrise, sunset, PROFILES["g_peak_clear"])
        # Serie de nubosidad: ruido blanco suavizado (paso bajo) -> nubes con
        # tiempo de residencia realista de varios minutos
        noise = rng.standard_normal(hours.size)
        kernel = np.ones(15) / 15.0
        smooth = np.convolve(noise, kernel, mode="same")
        smooth = (smooth - smooth.min()) / (np.ptp(smooth) + 1e-9)
        cloud = 1.0 - PROFILES["cloud_depth"] * smooth
        # Eventos de nube densa (caídas bruscas)
        drops = rng.random(hours.size) < 0.05
        cloud[drops] *= rng.uniform(0.35, 0.75, size=drops.sum())
        g = env * cloud * (peak / PROFILES["g_peak_clear"] + 0.35)
        g = np.clip(g, 0.0, PROFILES["g_peak_clear"])

    elif profile == "Día lluvioso":
        peak = g_peak or PROFILES["g_peak_rainy"]
        env = _clear_sky_envelope(hours, sunrise, sunset, peak)
        g = env * (1.0 + PROFILES["rain_noise"] * rng.standard_normal(hours.size))

    else:
        raise ValueError(f"Perfil de irradiancia desconocido: {profile!r}")

    return np.clip(g, 0.0, None)


# ===========================================================================
# Temperatura ambiente
# ===========================================================================
def generate_temperature(
    season: str,
    hours: np.ndarray,
    t_min: float | None = None,
    t_max: float | None = None,
    t_peak_h: float | None = None,
) -> np.ndarray:
    """
    Tamb(t) [°C]: sinusoide con mínimo al amanecer y máximo a t_peak_h.

        Tamb = Tmed - A * cos( 2*pi * (t - t_peak + 12) / 24 )
    """
    if season not in PROFILES["seasons"]:
        raise ValueError(f"Estación desconocida: {season!r}")
    cfg = PROFILES["seasons"][season]

    t_min = cfg["t_min"] if t_min is None else t_min
    t_max = cfg["t_max"] if t_max is None else t_max
    t_peak_h = cfg["t_peak_h"] if t_peak_h is None else t_peak_h

    t_mean = 0.5 * (t_max + t_min)
    amp = 0.5 * (t_max - t_min)
    return t_mean + amp * np.cos(2.0 * np.pi * (hours - t_peak_h) / 24.0)


# ===========================================================================
# Ensamblado de perfiles
# ===========================================================================
def build_synthetic_profile(
    irradiance_profile: str,
    season: str,
    timestep_min: int = PROFILES["timestep_min"],
    sunrise: float = PROFILES["sunrise_h"],
    sunset: float = PROFILES["sunset_h"],
    g_peak: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    seed: int | None = PROFILES["seed"],
) -> pd.DataFrame:
    """DataFrame con columnas [G, Tamb] indexado por timestamp."""
    idx, hours = time_axis(timestep_min)
    g = generate_irradiance(irradiance_profile, hours, sunrise, sunset, g_peak, seed)
    t_amb = generate_temperature(season, hours, t_min, t_max)

    df = pd.DataFrame({"G": g, "Tamb": t_amb}, index=idx)
    df.index.name = "timestamp"
    df["hour"] = hours
    return df


# ===========================================================================
# Carga de perfiles personalizados (CSV)
# ===========================================================================
def load_custom_profile(file_or_buffer) -> pd.DataFrame:
    """
    Carga un CSV con formato:

        timestamp,G,Tamb
        00:00,0,20
        ...
        12:00,950,32

    `timestamp` admite "HH:MM" o una fecha-hora completa.
    Devuelve un DataFrame validado con columnas [G, Tamb, hour].
    """
    df = pd.read_csv(file_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]

    required = [c.lower() for c in UI["csv_template_cols"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV: {missing}. "
            f"Formato esperado: {','.join(UI['csv_template_cols'])}"
        )

    ts = df["timestamp"].astype(str).str.strip()
    try:
        idx = pd.to_datetime(ts, format="%H:%M")
    except (ValueError, TypeError):
        idx = pd.to_datetime(ts)

    # IMPORTANTE: usar .to_numpy() -> si se pasan Series, pandas alinea por el
    # índice original (RangeIndex) contra el nuevo DatetimeIndex y genera NaN.
    out = pd.DataFrame(
        {
            "G": pd.to_numeric(df["g"], errors="coerce").to_numpy(),
            "Tamb": pd.to_numeric(df["tamb"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    ).sort_index()

    if out[["G", "Tamb"]].isna().any().any():
        raise ValueError("El CSV contiene valores no numéricos o vacíos en G / Tamb.")
    if (out["G"] < 0).any():
        raise ValueError("El CSV contiene irradiancias negativas.")

    out["hour"] = out.index.hour + out.index.minute / 60.0
    return out


def infer_timestep_hours(df: pd.DataFrame) -> float:
    """Paso temporal medio del perfil, en horas (para integrar energía)."""
    if len(df) < 2:
        return 1.0
    deltas = np.diff(df.index.values).astype("timedelta64[s]").astype(float)
    return float(np.median(deltas)) / 3600.0
