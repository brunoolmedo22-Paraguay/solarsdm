"""
app.py — PV Simulator (Single Diode Model)
==========================================
Interfaz Streamlit. Esta capa NO contiene física: sólo orquesta

    config/  (parámetros)  ->  models/ (física)  ->  simulation/ (resolución)
                                                 ->  visualization/ (gráficos)

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from config.pv_database import (
    CUSTOM_KEY,
    default_custom_stc,
    get_module,
    list_manufacturers,
    list_models,
)
from config.settings import G_REF, T_REF_C, UI, PROFILES, SOLVER, THERMAL
from models.irradiance_model import (
    IRRADIANCE_PROFILES,
    SEASONS,
    build_synthetic_profile,
    infer_timestep_hours,
    load_custom_profile,
)
from models.pv_module import PVModule, SDMParams
from models.single_diode import iv_curve
from models.temperature_model import cell_temperature_noct
from simulation.energy import compute_kpis
from simulation.mpp import find_mpp, simulate_timeseries
from simulation.solver import extract_sdm_params, translate_params
from visualization.plots import (
    plot_efficiency,
    plot_energy,
    plot_irradiance,
    plot_iv_family,
    plot_iv_pv,
    plot_mpp_trajectory,
    plot_power,
    plot_temperatures,
)

# ===========================================================================
# Configuración de la página
# ===========================================================================
st.set_page_config(
    page_title=UI["app_title"],
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      div[data-testid="stMetric"] {
          background: #FAFBFC; border: 1px solid #E6E9EE;
          border-radius: 10px; padding: 12px 14px;
      }
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {padding: 8px 16px;}
      .caption-box {
          background:#F5F8FA; border-left:3px solid #2E7D6B;
          padding:10px 14px; border-radius:6px; font-size:0.86rem; color:#3C4650;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Estado de la sesión
# ===========================================================================
def init_state():
    ss = st.session_state
    ss.setdefault("module", None)          # PVModule activo
    ss.setdefault("extraction", None)      # ExtractionReport
    ss.setdefault("profile", None)         # DataFrame [G, Tamb]
    ss.setdefault("profile_desc", "—")
    ss.setdefault("results", None)         # DataFrame simulado
    ss.setdefault("kpis", None)
    ss.setdefault("noct", THERMAL["default_noct"])


init_state()


@st.cache_data(show_spinner=False)
def cached_extraction(stc_dict: dict):
    """Extracción de los 5 parámetros SDM (cacheada por hoja de datos)."""
    from models.pv_module import ModuleSTC
    stc = ModuleSTC(**stc_dict)
    sdm, report = extract_sdm_params(stc)
    return sdm.to_dict(), report


# ===========================================================================
# Cabecera
# ===========================================================================
st.title("☀️ PV Simulator — Single Diode Model")
st.caption(
    "Simulación electro-térmica de módulos fotovoltaicos resolviendo la ecuación "
    "completa del circuito equivalente de un diodo. Sin aproximaciones lineales de potencia."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Configuración del panel",
    "2 · Irradiancia y temperatura",
    "3 · Modelo y simulación",
    "4 · Resultados",
])


# ===========================================================================
# TAB 1 — CONFIGURACIÓN DEL PANEL
# ===========================================================================
with tab1:
    st.subheader("Selección del módulo")

    c1, c2, c3 = st.columns([1.1, 1.4, 1.0])
    with c1:
        manufacturer = st.selectbox("Fabricante", list_manufacturers(), index=0)
    with c2:
        model_key = st.selectbox("Modelo", list_models(manufacturer), index=0)

    # --- Datos base -------------------------------------------------------
    if manufacturer == CUSTOM_KEY or model_key == CUSTOM_KEY:
        base_stc = default_custom_stc()
    else:
        base_stc = get_module(model_key).stc

    with c3:
        st.metric("Potencia de catálogo", f"{base_stc.p_nom:.0f} W")

    st.divider()

    # --- Parámetros eléctricos (editables) --------------------------------
    st.subheader("Parámetros eléctricos (STC · 1000 W/m², 25 °C, AM1.5)")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        v_oc = st.number_input("Voc [V]", 5.0, 200.0, float(base_stc.v_oc), 0.1, format="%.2f")
        i_sc = st.number_input("Isc [A]", 0.1, 40.0, float(base_stc.i_sc), 0.01, format="%.2f")
    with e2:
        v_mp = st.number_input("Vmp [V]", 4.0, 190.0, float(base_stc.v_mp), 0.1, format="%.2f")
        i_mp = st.number_input("Imp [A]", 0.1, 40.0, float(base_stc.i_mp), 0.01, format="%.2f")
    with e3:
        p_nom = st.number_input("Pnom [W]", 10.0, 1000.0, float(base_stc.p_nom), 1.0, format="%.1f")
        n_cells = st.number_input("Nº células serie (Ns)", 12, 200, int(base_stc.n_cells), 1)
    with e4:
        length = st.number_input("Largo [m]", 0.3, 3.0, float(base_stc.length), 0.001, format="%.3f")
        width = st.number_input("Ancho [m]", 0.3, 2.0, float(base_stc.width), 0.001, format="%.3f")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        alpha = st.number_input("α Isc [%/°C]", -0.5, 0.5, float(base_stc.alpha_isc_pct),
                                0.001, format="%.4f")
    with t2:
        beta = st.number_input("β Voc [%/°C]", -1.0, 0.0, float(base_stc.beta_voc_pct),
                               0.001, format="%.4f")
    with t3:
        gamma = st.number_input("γ Pmax [%/°C]", -1.0, 0.0, float(base_stc.gamma_pmax_pct),
                                0.001, format="%.4f")
    with t4:
        noct = st.number_input("NOCT [°C]", 30.0, 60.0, float(base_stc.noct), 0.5, format="%.1f")

    stc = replace(
        base_stc,
        p_nom=p_nom, v_oc=v_oc, i_sc=i_sc, v_mp=v_mp, i_mp=i_mp,
        n_cells=int(n_cells), alpha_isc_pct=alpha, beta_voc_pct=beta,
        gamma_pmax_pct=gamma, noct=noct, length=length, width=width,
    )
    st.session_state["noct"] = noct

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Área bruta", f"{stc.area:.3f} m²")
    m2.metric("Eficiencia STC", f"{stc.efficiency_stc*100:.2f} %")
    m3.metric("Vmp · Imp", f"{stc.p_mp_datasheet:.1f} W")
    m4.metric("Fill Factor (cat.)", f"{stc.p_mp_datasheet/(stc.v_oc*stc.i_sc):.3f}")

    st.divider()

    # --- Parámetros SDM ---------------------------------------------------
    st.subheader("Parámetros del Single Diode Model")
    st.markdown(
        '<div class="caption-box">Los fabricantes no publican <b>IL, I0, Rs, Rsh, n</b>. '
        "Se estiman resolviendo el sistema no lineal de 5 ecuaciones de De Soto et al. (2006) "
        "sobre los puntos de catálogo (Isc, Voc, MPP, dP/dV = 0 y β<sub>Voc</sub>). "
        "Todos los valores son editables.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        do_extract = st.button("⚙️ Estimar parámetros SDM", type="primary", width="stretch")

    if do_extract or st.session_state["module"] is None:
        with st.spinner("Resolviendo el sistema de 5 ecuaciones…"):
            sdm_dict, report = cached_extraction(stc.to_dict())
        st.session_state["module"] = PVModule(stc=stc, sdm=SDMParams(**sdm_dict))
        st.session_state["extraction"] = report

    module: PVModule = st.session_state["module"]
    module.stc = stc                                # mantener STC sincronizado
    sdm = module.sdm
    report = st.session_state["extraction"]

    with col_msg:
        if report is not None:
            if report.success:
                st.success(
                    f"Ajuste convergido · residuo cuadrático = {report.cost:.2e} "
                    f"· {report.n_iter} evaluaciones"
                )
            else:
                st.warning(f"Ajuste con residuo alto ({report.cost:.2e}). Revisá los datos STC.")

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        il = st.number_input("Iph = IL [A]", 0.0, 40.0, float(sdm.IL_ref), 0.0001, format="%.4f")
    with s2:
        i0_exp = st.number_input("log₁₀(I0 [A])", -16.0, -3.0,
                                 float(np.log10(sdm.I0_ref)), 0.01, format="%.3f")
    with s3:
        rs = st.number_input("Rs [Ω]", 0.0001, 5.0, float(sdm.Rs), 0.0001, format="%.4f")
    with s4:
        rsh = st.number_input("Rsh [Ω]", 10.0, 100000.0, float(sdm.Rsh_ref), 1.0, format="%.1f")
    with s5:
        n_id = st.number_input("n (idealidad)", 0.5, 2.5, float(sdm.n), 0.001, format="%.4f")

    module.sdm = SDMParams(
        IL_ref=il, I0_ref=10.0**i0_exp, Rs=rs, Rsh_ref=rsh, n=n_id,
        n_cells=int(n_cells), source="manual" if not do_extract else sdm.source,
    )
    st.session_state["module"] = module

    st.caption(
        f"I0 = {module.sdm.I0_ref:.4e} A  ·  "
        f"a_ref = n·Ns·k·T/q = {module.sdm.a_ref:.4f} V  ·  "
        f"Vt (célula, 25 °C) = {module.sdm.vt_ref*1000:.3f} mV"
    )

    if report is not None:
        with st.expander("Residuos del ajuste (normalizados)"):
            st.dataframe(
                pd.DataFrame(
                    {"residuo": report.residuals}
                ).style.format("{:.3e}"),
                width="stretch",
            )

    # --- Verificación instantánea en STC ----------------------------------
    st.divider()
    st.subheader("Verificación del modelo en STC")
    p_stc = translate_params(module.sdm, stc, G_REF, T_REF_C)
    mpp_stc = find_mpp(p_stc)
    check = pd.DataFrame({
        "Catálogo": [stc.i_sc, stc.v_oc, stc.v_mp, stc.i_mp, stc.p_nom],
        "Modelo SDM": [mpp_stc["Isc"], mpp_stc["Voc"], mpp_stc["Vmp"],
                       mpp_stc["Imp"], mpp_stc["Pmp"]],
    }, index=["Isc [A]", "Voc [V]", "Vmp [V]", "Imp [A]", "Pmax [W]"])
    check["Error [%]"] = (check["Modelo SDM"] - check["Catálogo"]) / check["Catálogo"] * 100

    v1, v2 = st.columns([1.1, 1.5])
    with v1:
        st.dataframe(check.style.format("{:.3f}"), width="stretch")
    with v2:
        V, I, P = iv_curve(p_stc)
        st.plotly_chart(
            plot_iv_pv([{"label": "STC", "V": V, "I": I, "P": P, "mpp": mpp_stc}],
                       "· condiciones STC"),
            width="stretch",
        )


# ===========================================================================
# TAB 2 — IRRADIANCIA Y TEMPERATURA
# ===========================================================================
with tab2:
    st.subheader("Perfil de irradiancia y temperatura")

    mode = st.radio(
        "Modo",
        ["MODO B · Generador de perfiles característicos", "MODO A · Carga personalizada (CSV)"],
        horizontal=True,
    )

    # ------------------------- MODO A -------------------------------------
    if mode.startswith("MODO A"):
        st.markdown(
            '<div class="caption-box">Formato esperado: <code>timestamp,G,Tamb</code> · '
            "resolución recomendada 1 minuto · timestamp como <code>HH:MM</code> o fecha-hora "
            "completa. G en W/m², Tamb en °C.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        cA, cB = st.columns([2, 1])
        with cA:
            up = st.file_uploader("Archivo CSV", type=["csv"])
        with cB:
            template = pd.DataFrame({
                "timestamp": ["00:00", "06:00", "12:00", "18:00", "23:59"],
                "G": [0, 120, 950, 90, 0],
                "Tamb": [20.0, 22.0, 32.0, 26.0, 21.0],
            })
            st.download_button(
                "⬇️ Plantilla CSV",
                template.to_csv(index=False).encode("utf-8"),
                file_name="plantilla_perfil.csv",
                mime="text/csv",
                width="stretch",
            )

        if up is not None:
            try:
                df = load_custom_profile(up)
                st.session_state["profile"] = df
                st.session_state["profile_desc"] = f"CSV · {up.name} · {len(df)} registros"
                st.success(f"Perfil cargado: {len(df)} registros "
                           f"(Δt ≈ {infer_timestep_hours(df)*60:.1f} min)")
            except Exception as exc:
                st.error(f"Error al leer el CSV: {exc}")

    # ------------------------- MODO B -------------------------------------
    else:
        g1, g2, g3 = st.columns(3)
        with g1:
            prof = st.selectbox("Perfil de irradiancia", IRRADIANCE_PROFILES)
        with g2:
            season = st.selectbox("Estación (temperatura)", SEASONS)
        with g3:
            dt_min = st.select_slider("Paso temporal [min]", [1, 5, 10, 15, 30, 60], value=1)

        with st.expander("Ajustes finos del perfil"):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                sunrise = st.number_input("Amanecer [h]", 3.0, 9.0,
                                          float(PROFILES["sunrise_h"]), 0.25)
            with f2:
                sunset = st.number_input("Atardecer [h]", 15.0, 22.0,
                                         float(PROFILES["sunset_h"]), 0.25)
            with f3:
                cfg = PROFILES["seasons"][season]
                t_min_v = st.number_input("T mín [°C]", -20.0, 40.0, float(cfg["t_min"]), 0.5)
            with f4:
                t_max_v = st.number_input("T máx [°C]", -10.0, 55.0, float(cfg["t_max"]), 0.5)

            g_peak = st.slider("Irradiancia pico [W/m²]", 100, 1200,
                               int({"Día soleado": PROFILES["g_peak_clear"],
                                    "Día nublado": PROFILES["g_peak_cloudy"],
                                    "Día lluvioso": PROFILES["g_peak_rainy"]}[prof]), 10)
            seed = st.number_input("Semilla aleatoria", 0, 9999, int(PROFILES["seed"]), 1)

        if st.button("🔄 Generar perfil", type="primary"):
            df = build_synthetic_profile(
                prof, season, timestep_min=int(dt_min),
                sunrise=sunrise, sunset=sunset, g_peak=float(g_peak),
                t_min=t_min_v, t_max=t_max_v, seed=int(seed),
            )
            st.session_state["profile"] = df
            st.session_state["profile_desc"] = (
                f"{prof} · {season} · Δt = {dt_min} min · Gpico = {g_peak} W/m²"
            )
            st.session_state["results"] = None

    # ------------------------- Modelo térmico -----------------------------
    st.divider()
    st.subheader("Modelo térmico")
    th1, th2 = st.columns([1, 3])
    with th1:
        noct_ui = st.number_input(
            "NOCT [°C]", 30.0, 60.0, float(st.session_state["noct"]), 0.5,
            help="Nominal Operating Cell Temperature (ensayo: 800 W/m², 20 °C, 1 m/s).",
        )
        st.session_state["noct"] = noct_ui
    with th2:
        st.latex(r"T_c = T_{amb} + \frac{NOCT - 20}{800}\, G")
        st.caption(
            f"Pendiente actual: {(noct_ui-20)/800:.5f} °C por W/m²  →  "
            f"a 1000 W/m² la célula está {((noct_ui-20)/800)*1000:.1f} °C por encima del ambiente."
        )

    # ------------------------- Vista previa -------------------------------
    df = st.session_state["profile"]
    if df is not None:
        df = df.copy()
        df["Tc"] = cell_temperature_noct(df["Tamb"].to_numpy(), df["G"].to_numpy(), noct_ui)
        st.session_state["profile"] = df

        p1, p2, p3, p4 = st.columns(4)
        dt_h = infer_timestep_hours(df)
        p1.metric("G máxima", f"{df['G'].max():.0f} W/m²")
        p2.metric("Irradiación diaria", f"{df['G'].sum()*dt_h/1000:.2f} kWh/m²")
        p3.metric("T amb. máxima", f"{df['Tamb'].max():.1f} °C")
        p4.metric("T célula máxima", f"{df['Tc'].max():.1f} °C")

        st.plotly_chart(plot_irradiance(df), width="stretch")
        st.plotly_chart(plot_temperatures(df), width="stretch")
    else:
        st.info("Generá un perfil sintético o cargá un CSV para continuar.")


# ===========================================================================
# TAB 3 — MODELO Y SIMULACIÓN
# ===========================================================================
with tab3:
    module = st.session_state["module"]
    profile = st.session_state["profile"]

    if module is None:
        st.warning("Configurá primero el módulo en la pestaña 1.")
    else:
        st.subheader("Resumen de la simulación")

        i1, i2 = st.columns(2)
        with i1:
            st.markdown("**Módulo**")
            st.dataframe(pd.DataFrame({
                "Valor": {
                    "Fabricante": module.stc.manufacturer,
                    "Modelo": module.stc.model,
                    "Tecnología": module.stc.technology,
                    "Pnom [W]": f"{module.stc.p_nom:.1f}",
                    "Ns (células serie)": str(module.stc.n_cells),
                    "Área [m²]": f"{module.stc.area:.3f}",
                    "η STC [%]": f"{module.stc.efficiency_stc*100:.2f}",
                    "NOCT [°C]": f"{st.session_state['noct']:.1f}",
                }
            }), width="stretch")
        with i2:
            st.markdown("**Parámetros SDM utilizados (STC)**")
            s = module.sdm
            st.dataframe(pd.DataFrame({
                "Valor": {
                    "IL_ref [A]": f"{s.IL_ref:.4f}",
                    "I0_ref [A]": f"{s.I0_ref:.4e}",
                    "Rs [Ω]": f"{s.Rs:.4f}",
                    "Rsh_ref [Ω]": f"{s.Rsh_ref:.1f}",
                    "n [-]": f"{s.n:.4f}",
                    "a_ref [V]": f"{s.a_ref:.4f}",
                    "Origen": s.source,
                }
            }), width="stretch")

        st.divider()
        st.subheader("Condiciones de contorno")

        if profile is None:
            st.warning("No hay perfil cargado. Volvé a la pestaña 2.")
        else:
            dt_h = infer_timestep_hours(profile)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Perfil", st.session_state["profile_desc"].split("·")[0].strip())
            c2.metric("Duración", f"{len(profile)*dt_h:.1f} h")
            c3.metric("Paso temporal", f"{dt_h*60:.0f} min")
            c4.metric("Nº de pasos", f"{len(profile)}")
            st.caption(f"Detalle del perfil: {st.session_state['profile_desc']}")

            st.divider()
            st.subheader("Configuración del generador")
            a1, a2, a3 = st.columns(3)
            with a1:
                n_series = st.number_input("Módulos en serie (string)", 1, 40, 1)
            with a2:
                n_parallel = st.number_input("Strings en paralelo", 1, 100, 1)
            with a3:
                soiling = st.slider("Pérdidas por suciedad [%]", 0.0, 15.0, 0.0, 0.5) / 100.0

            n_mod = int(n_series) * int(n_parallel)
            st.caption(
                f"Generador: **{n_mod} módulo(s)** · "
                f"Potencia instalada = **{module.stc.p_nom*n_mod/1000:.3f} kWp** · "
                f"Área = **{module.stc.area*n_mod:.2f} m²**  ·  "
                "Hipótesis: MPPT ideal (el sistema opera siempre en el MPP)."
            )

            st.divider()
            st.markdown(
                '<div class="caption-box">Para <b>cada paso temporal</b>: '
                "1) leer G  →  2) leer Tamb  →  3) calcular Tc (NOCT)  →  "
                "4) trasladar los 5 parámetros a (G, Tc)  →  "
                "5) resolver la curva I-V/P-V del SDM  →  "
                "6) localizar numéricamente el MPP  →  7) guardar Vmp, Imp, Pmp, η.</div>",
                unsafe_allow_html=True,
            )
            st.write("")

            if st.button("▶️  SIMULAR", type="primary", width="stretch"):
                bar = st.progress(0.0, text="Resolviendo el Single Diode Model…")
                res = simulate_timeseries(
                    module, profile, noct=st.session_state["noct"],
                    n_series=int(n_series), n_parallel=int(n_parallel),
                    soiling_losses=float(soiling),
                    progress_callback=lambda f: bar.progress(
                        min(f, 1.0), text=f"Resolviendo el SDM… {f*100:.0f} %"),
                )
                bar.empty()
                st.session_state["results"] = res
                st.session_state["kpis"] = compute_kpis(res, module, infer_timestep_hours(res))
                st.success(
                    f"Simulación completada: {len(res)} puntos de operación resueltos. "
                    "Los resultados están en la pestaña 4."
                )


# ===========================================================================
# TAB 4 — RESULTADOS
# ===========================================================================
with tab4:
    res = st.session_state["results"]
    kpi = st.session_state["kpis"]
    module = st.session_state["module"]

    if res is None or kpi is None:
        st.info("Ejecutá una simulación en la pestaña 3 para ver los resultados.")
    else:
        dt_h = kpi["dt_hours"]

        # ------------------------- KPIs ----------------------------------
        st.subheader("Indicadores del sistema")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Energía diaria", f"{kpi['E_day_kWh']:.3f} kWh")
        k2.metric("Yield específico", f"{kpi['specific_yield_kWh_kWp']:.2f} kWh/kWp")
        k3.metric("Performance Ratio", f"{kpi['PR']:.3f}")
        k4.metric("Factor de capacidad", f"{kpi['CF']*100:.2f} %")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Potencia máxima", f"{kpi['P_max_W']:.1f} W",
                  help="Salida en el MPP resuelta con el SDM.")
        k6.metric("Horario del pico",
                  kpi["t_peak"].strftime("%H:%M") if kpi["t_peak"] is not None else "—")
        k7.metric("Eficiencia promedio", f"{kpi['eta_mean']*100:.2f} %",
                  delta=f"{(kpi['eta_mean']-kpi['eta_stc'])*100:+.2f} pp vs STC")
        k8.metric("Potencia media (día)", f"{kpi['P_mean_W']:.1f} W")

        k9, k10, k11, k12 = st.columns(4)
        k9.metric("Horas equivalentes", f"{kpi['equivalent_hours_h']:.2f} h")
        k10.metric("Irradiación (PSH)", f"{kpi['PSH_h']:.2f} kWh/m²")
        k11.metric("Energía solar incidente", f"{kpi['E_solar_kWh']:.2f} kWh")
        k12.metric("Tc máxima", f"{kpi['Tc_max_C']:.1f} °C",
                   delta=f"{kpi['Tc_mean_sun_C']:.1f} °C media (con sol)")

        st.divider()

        # ------------------------- Series temporales ----------------------
        st.subheader("Series temporales")
        st.plotly_chart(plot_irradiance(res), width="stretch")
        st.plotly_chart(plot_temperatures(res), width="stretch")

        show_ref = st.checkbox(
            "Superponer la referencia lineal P = Pnom·G/1000 (solo comparación)", value=False,
            help="El simulador NO usa esta expresión. Se muestra únicamente para "
                 "cuantificar el error de esa aproximación frente al SDM.",
        )
        st.plotly_chart(plot_power(res, show_linear_ref=show_ref), width="stretch")

        if show_ref:
            err = (kpi["E_lineal_ref_kWh"] - kpi["E_day_kWh"]) / kpi["E_day_kWh"] * 100
            st.caption(
                f"La aproximación lineal predice {kpi['E_lineal_ref_kWh']:.3f} kWh frente a los "
                f"{kpi['E_day_kWh']:.3f} kWh del SDM → sobrestima la energía en **{err:+.1f} %** "
                "(ignora el derating térmico y las pérdidas resistivas)."
            )

        st.plotly_chart(plot_efficiency(res, module.stc.efficiency_stc), width="stretch")
        st.plotly_chart(plot_energy(res, dt_h), width="stretch")

        st.divider()

        # ------------------------- Curvas I-V / P-V -----------------------
        st.subheader("Curvas I-V y P-V en condiciones seleccionadas")

        cv1, cv2, cv3 = st.columns([1, 1, 1.2])
        with cv1:
            hours = sorted(res.index.strftime("%H:%M").tolist())
            default_i = int(res["Pmp"].to_numpy().argmax())
            sel = st.select_slider(
                "Instante del día", options=hours,
                value=res.index[default_i].strftime("%H:%M"),
            )
        row = res[res.index.strftime("%H:%M") == sel].iloc[0]
        with cv2:
            st.metric("G / Tc en el instante", f"{row['G']:.0f} W/m²  ·  {row['Tc']:.1f} °C")
        with cv3:
            st.metric("MPP resuelto",
                      f"{row['Pmp']:.1f} W  ·  {row['Vmp']:.1f} V  ·  {row['Imp']:.2f} A")

        curves = []
        if row["G"] > SOLVER["g_min"]:
            p_op = translate_params(module.sdm, module.stc, row["G_eff"], row["Tc"])
            V, I, P = iv_curve(p_op)
            curves.append({
                "label": f"{sel} · {row['G']:.0f} W/m², {row['Tc']:.0f} °C",
                "V": V, "I": I, "P": P,
                "mpp": {"Vmp": row["Vmp"], "Imp": row["Imp"], "Pmp": row["Pmp"]},
            })
        p_stc = translate_params(module.sdm, module.stc, G_REF, T_REF_C)
        Vs, Is, Ps = iv_curve(p_stc)
        curves.append({"label": "STC (referencia)", "V": Vs, "I": Is, "P": Ps,
                       "mpp": find_mpp(p_stc), "color": "#9AA5B1"})

        st.plotly_chart(plot_iv_pv(curves), width="stretch")

        # --- Familias de curvas (validación física visual) -----------------
        with st.expander("Familias de curvas I-V (validación física del modelo)"):
            fam1, fam2 = st.columns(2)
            with fam1:
                fam_g = {}
                for g in (200, 400, 600, 800, 1000):
                    p = translate_params(module.sdm, module.stc, g, 25.0)
                    V, I, _ = iv_curve(p, n_points=200)
                    fam_g[f"{g} W/m²"] = {"V": V, "I": I, "mpp": find_mpp(p)}
                st.plotly_chart(
                    plot_iv_family(fam_g, "Tensión V [V]",
                                   "Efecto de la irradiancia (Tc = 25 °C) — la corriente escala con G"),
                    width="stretch",
                )
            with fam2:
                fam_t = {}
                for t in (15, 25, 45, 65):
                    p = translate_params(module.sdm, module.stc, G_REF, float(t))
                    V, I, _ = iv_curve(p, n_points=200)
                    fam_t[f"{t} °C"] = {"V": V, "I": I, "mpp": find_mpp(p)}
                st.plotly_chart(
                    plot_iv_family(fam_t, "Tensión V [V]",
                                   "Efecto de la temperatura (G = 1000 W/m²) — Voc cae con Tc"),
                    width="stretch",
                )

        st.plotly_chart(plot_mpp_trajectory(res), width="stretch")

        st.divider()

        # ------------------------- Descargas ------------------------------
        st.subheader("Exportación")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Serie temporal simulada (CSV)",
                res.to_csv().encode("utf-8"),
                file_name="pv_simulacion_series.csv",
                mime="text/csv",
                width="stretch",
            )
        with d2:
            kpi_out = {k: (v.strftime("%H:%M") if hasattr(v, "strftime") else v)
                       for k, v in kpi.items()}
            st.download_button(
                "⬇️ KPIs (CSV)",
                pd.Series(kpi_out, name="valor").to_csv().encode("utf-8"),
                file_name="pv_simulacion_kpis.csv",
                mime="text/csv",
                width="stretch",
            )

        with st.expander("Tabla de resultados (primeras 200 filas)"):
            st.dataframe(res.head(200), width="stretch")


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown("### Estado")
    mod = st.session_state["module"]
    st.write("**Módulo:**", mod.name if mod else "—")
    st.write("**Perfil:**", st.session_state["profile_desc"])
    st.write("**Simulación:**",
             "✅ resuelta" if st.session_state["results"] is not None else "⏳ pendiente")

    st.divider()
    st.markdown("### Modelo")
    st.latex(r"I = I_{ph} - I_0\left[e^{\frac{V+IR_s}{n N_s V_t}}-1\right] - \frac{V+IR_s}{R_{sh}}")
    st.caption(
        "Traslación a (G, Tc) según De Soto et al. (2006). "
        "Resolución: función W de Lambert (exacta) con verificación cruzada "
        "Brent / Newton-Raphson. MPP por minimización acotada de -P(V)."
    )

    st.divider()
    st.caption("Hipótesis: MPPT ideal → Pout(t) = Pmp(t). No se modela el inversor.")
