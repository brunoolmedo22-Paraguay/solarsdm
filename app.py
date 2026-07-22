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
    detect_profile_columns,
    infer_timestep_hours,
    prepare_custom_profile,
    read_custom_profile_table,
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
    plot_profile_coverage,
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
    ss.setdefault("profile_meta", None)
    ss.setdefault("uploaded_profile_name", None)
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


def select_date_window(df: pd.DataFrame, key_prefix: str) -> tuple[pd.DataFrame, str]:
    """Selector reutilizable para mostrar un día o un intervalo de fechas."""
    if df is None or df.empty:
        return df, "—"

    min_date = df.index.min().date()
    max_date = df.index.max().date()
    if min_date == max_date:
        return df, min_date.isoformat()

    mode = st.radio(
        "Período mostrado", ["Um dia", "Intervalo de dias"],
        horizontal=True, key=f"{key_prefix}_mode",
    )
    if mode == "Um dia":
        selected = st.date_input(
            "Dia", value=min_date, min_value=min_date, max_value=max_date,
            key=f"{key_prefix}_day",
        )
        start = pd.Timestamp(selected)
        end = start + pd.Timedelta(days=1)
        label = selected.isoformat()
    else:
        selected = st.date_input(
            "Intervalo", value=(min_date, max_date),
            min_value=min_date, max_value=max_date, key=f"{key_prefix}_range",
        )
        if isinstance(selected, (tuple, list)) and len(selected) == 2:
            start = pd.Timestamp(selected[0])
            end = pd.Timestamp(selected[1]) + pd.Timedelta(days=1)
        else:
            start = pd.Timestamp(min_date)
            end = pd.Timestamp(max_date) + pd.Timedelta(days=1)
        label = f"{start.date().isoformat()} a {(end - pd.Timedelta(days=1)).date().isoformat()}"

    view = df.loc[(df.index >= start) & (df.index < end)].copy()
    return view, label


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
            key="chart_iv_pv_stc",
        )


# ===========================================================================
# TAB 2 — IRRADIANCIA Y TEMPERATURA
# ===========================================================================
with tab2:
    st.subheader("Perfil de irradiância e temperatura")

    mode = st.radio(
        "Modo",
        ["MODO B · Gerador de perfis característicos", "MODO A · Carregamento personalizado (CSV)"],
        horizontal=True,
    )

    # ------------------------- MODO A -------------------------------------
    if mode.startswith("MODO A"):
        st.markdown(
            '<div class="caption-box">O carregador reconhece nomes como '
            '<code>Timestamp</code>, <code>G</code>, <code>GHI_REAL</code>, '
            '<code>GHI_PREDITO</code> e colunas de temperatura. A série escolhida '
            'é convertida internamente para <code>G</code> [W/m²]. Os intervalos '
            'ausentes podem ser completados com G = 0 e ficam identificados.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        cA, cB = st.columns([2, 1])
        with cA:
            up = st.file_uploader("Arquivo CSV", type=["csv"], key="profile_csv")
        with cB:
            template = pd.DataFrame({
                "Timestamp": ["2026-01-01 06:00:00", "2026-01-01 12:00:00", "2026-01-01 18:00:00"],
                "GHI_PREDITO": [0.0, 950.0, 0.0],
                "Temperatura_C": [22.0, 31.0, 26.0],
            })
            st.download_button(
                "⬇️ Modelo de CSV",
                template.to_csv(index=False).encode("utf-8"),
                file_name="modelo_perfil_previsao.csv",
                mime="text/csv",
                width="stretch",
            )

        if up is not None:
            if st.session_state["uploaded_profile_name"] != up.name:
                st.session_state["uploaded_profile_name"] = up.name
                st.session_state["profile"] = None
                st.session_state["profile_meta"] = None
                st.session_state["results"] = None

            try:
                raw = read_custom_profile_table(up)
                detected = detect_profile_columns(raw.columns)

                st.caption(
                    f"Arquivo detectado: **{len(raw):,} linhas** · "
                    f"{len(raw.columns)} colunas · {', '.join(map(str, raw.columns))}"
                )

                col1, col2, col3 = st.columns(3)
                columns = list(raw.columns)
                with col1:
                    ts_default = columns.index(detected["timestamp_default"]) if detected["timestamp_default"] in columns else 0
                    timestamp_col = st.selectbox(
                        "Coluna de timestamp", columns, index=ts_default,
                        key="csv_timestamp_col",
                    )
                with col2:
                    irr_options = detected["irradiance_candidates"] or columns
                    irr_default = detected["irradiance_default"]
                    irr_idx = irr_options.index(irr_default) if irr_default in irr_options else 0
                    irradiance_col = st.selectbox(
                        "Coluna de irradiância usada pelo modelo", irr_options, index=irr_idx,
                        help="Quando existem GHI_REAL e GHI_PREDITO, a previsão é selecionada por padrão.",
                        key="csv_irradiance_col",
                    )
                with col3:
                    none_temp = "— Temperatura não especificada —"
                    temp_options = [none_temp] + columns
                    temp_default = detected["temperature_default"]
                    temp_idx = temp_options.index(temp_default) if temp_default in temp_options else 0
                    temp_selection = st.selectbox(
                        "Coluna de temperatura ambiente", temp_options, index=temp_idx,
                        key="csv_temperature_col",
                    )
                    temperature_col = None if temp_selection == none_temp else temp_selection

                complete_days = st.checkbox(
                    "Completar todos os dias entre 00:00 e 23:59 com G = 0 nas lacunas",
                    value=True,
                    help=(
                        "A plataforma preserva uma coluna de rastreabilidade. Lacunas fora da "
                        "janela solar são marcadas como noite preenchida; lacunas dentro da "
                        "janela solar são destacadas separadamente para revisão."
                    ),
                )

                temp_strategy = "constant_day_night"
                temp_day, temp_night = 30.0, 20.0
                season = "Otoño/Primavera"
                t_min_v = t_max_v = None

                if temperature_col is None:
                    st.warning(
                        "Temperatura não especificada no CSV. Escolha como a temperatura ambiente "
                        "será gerada para a simulação."
                    )
                    temp_mode = st.radio(
                        "Temperatura ambiente",
                        ["Definir temperatura de dia e de noite", "Usar curva de temperatura padrão"],
                        horizontal=True,
                        key="csv_temp_mode",
                    )
                    if temp_mode.startswith("Definir"):
                        t1, t2 = st.columns(2)
                        with t1:
                            temp_day = st.number_input(
                                "Temperatura durante o dia [°C]", -30.0, 60.0, 30.0, 0.5,
                                key="csv_temp_day",
                            )
                        with t2:
                            temp_night = st.number_input(
                                "Temperatura durante a noite [°C]", -30.0, 60.0, 20.0, 0.5,
                                key="csv_temp_night",
                            )
                        temp_strategy = "constant_day_night"
                    else:
                        temp_strategy = "standard_curve"
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            season = st.selectbox("Curva padrão", SEASONS, index=2, key="csv_temp_season")
                        cfg = PROFILES["seasons"][season]
                        with c2:
                            t_min_v = st.number_input(
                                "Temperatura mínima [°C]", -30.0, 50.0,
                                float(cfg["t_min"]), 0.5, key="csv_temp_min",
                            )
                        with c3:
                            t_max_v = st.number_input(
                                "Temperatura máxima [°C]", -20.0, 60.0,
                                float(cfg["t_max"]), 0.5, key="csv_temp_max",
                            )

                if st.button("✅ Processar CSV", type="primary", width="stretch"):
                    with st.spinner("Reconstruindo o eixo temporal e validando os dados…"):
                        df, meta = prepare_custom_profile(
                            raw,
                            timestamp_col=timestamp_col,
                            irradiance_col=irradiance_col,
                            temperature_col=temperature_col,
                            complete_days=complete_days,
                            temperature_strategy=temp_strategy,
                            temp_day=float(temp_day),
                            temp_night=float(temp_night),
                            season=season,
                            t_min=t_min_v,
                            t_max=t_max_v,
                        )
                    st.session_state["profile"] = df
                    st.session_state["profile_meta"] = meta
                    st.session_state["profile_desc"] = (
                        f"CSV · {up.name} · G = {irradiance_col} · "
                        f"{meta['total_rows']} registros · Δt = {meta['timestep_minutes']:.1f} min"
                    )
                    st.session_state["results"] = None
                    st.success(
                        f"Perfil preparado: {meta['original_rows']:,} registros do CSV + "
                        f"{meta['filled_rows']:,} registros preenchidos."
                    )
            except Exception as exc:
                st.error(f"Erro ao ler ou processar o CSV: {exc}")

    # ------------------------- MODO B -------------------------------------
    else:
        g1, g2, g3 = st.columns(3)
        with g1:
            prof = st.selectbox("Perfil de irradiância", IRRADIANCE_PROFILES)
        with g2:
            season = st.selectbox("Estação (temperatura)", SEASONS)
        with g3:
            dt_min = st.select_slider("Passo temporal [min]", [1, 5, 10, 15, 30, 60], value=1)

        with st.expander("Ajustes finos do perfil"):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                sunrise = st.number_input("Amanhecer [h]", 3.0, 9.0,
                                          float(PROFILES["sunrise_h"]), 0.25)
            with f2:
                sunset = st.number_input("Pôr do sol [h]", 15.0, 22.0,
                                         float(PROFILES["sunset_h"]), 0.25)
            with f3:
                cfg = PROFILES["seasons"][season]
                t_min_v = st.number_input("T mín [°C]", -20.0, 40.0, float(cfg["t_min"]), 0.5)
            with f4:
                t_max_v = st.number_input("T máx [°C]", -10.0, 55.0, float(cfg["t_max"]), 0.5)

            g_peak = st.slider("Irradiância de pico [W/m²]", 100, 1200,
                               int({"Día soleado": PROFILES["g_peak_clear"],
                                    "Día nublado": PROFILES["g_peak_cloudy"],
                                    "Día lluvioso": PROFILES["g_peak_rainy"]}[prof]), 10)
            seed = st.number_input("Semente aleatória", 0, 9999, int(PROFILES["seed"]), 1)

        if st.button("🔄 Gerar perfil", type="primary"):
            df = build_synthetic_profile(
                prof, season, timestep_min=int(dt_min),
                sunrise=sunrise, sunset=sunset, g_peak=float(g_peak),
                t_min=t_min_v, t_max=t_max_v, seed=int(seed),
            )
            st.session_state["profile"] = df
            st.session_state["profile_meta"] = None
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
            help="Nominal Operating Cell Temperature (ensaio: 800 W/m², 20 °C, 1 m/s).",
        )
        st.session_state["noct"] = noct_ui
    with th2:
        st.latex(r"T_c = T_{amb} + \frac{NOCT - 20}{800}\, G")
        st.caption(
            f"Inclinação atual: {(noct_ui-20)/800:.5f} °C por W/m²  →  "
            f"a 1000 W/m² a célula fica {((noct_ui-20)/800)*1000:.1f} °C acima do ambiente."
        )

    # ------------------------- Vista prévia -------------------------------
    df = st.session_state["profile"]
    if df is not None:
        df = df.copy()
        df["Tc"] = cell_temperature_noct(df["Tamb"].to_numpy(), df["G"].to_numpy(), noct_ui)
        st.session_state["profile"] = df

        meta = st.session_state.get("profile_meta")
        if meta is not None:
            st.divider()
            st.subheader("Rastreabilidade do preenchimento")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Registros originais", f"{meta['original_rows']:,}")
            q2.metric("Noite preenchida", f"{meta['filled_night_rows']:,}")
            q3.metric("Lacunas diurnas", f"{meta['filled_day_gap_rows']:,}")
            q4.metric("Total após completar", f"{meta['total_rows']:,}")
            st.caption(
                f"Temperatura: {meta['temperature_source']} · "
                f"Período: {meta['start']} até {meta['end']}"
            )
            if meta["filled_day_gap_rows"] > 0:
                st.warning(
                    f"Foram encontrados {meta['filled_day_gap_rows']:,} timestamps ausentes dentro "
                    "da janela solar estimada. Eles foram preenchidos com G = 0, mas aparecem em "
                    "vermelho no mapa para não serem confundidos com noite."
                )
            st.plotly_chart(plot_profile_coverage(df), width="stretch", key="chart_profile_coverage")

        st.divider()
        st.subheader("Visualização do período")
        df_view, period_label = select_date_window(df, "profile_preview")
        if df_view.empty:
            st.warning("O intervalo selecionado não contém dados.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            dt_h = infer_timestep_hours(df_view)
            p1.metric("G máxima", f"{df_view['G'].max():.0f} W/m²")
            p2.metric("Irradiação no período", f"{df_view['G'].sum()*dt_h/1000:.2f} kWh/m²")
            p3.metric("T amb. máxima", f"{df_view['Tamb'].max():.1f} °C")
            p4.metric("T célula máxima", f"{df_view['Tc'].max():.1f} °C")
            st.caption(f"Período exibido: {period_label} · {len(df_view):,} registros")

            st.plotly_chart(plot_irradiance(df_view), width="stretch", key="chart_irr_preview")
            st.plotly_chart(plot_temperatures(df_view), width="stretch", key="chart_temp_preview")
    else:
        st.info("Gere um perfil sintético ou carregue e processe um CSV para continuar.")


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
                st.success(
                    f"Simulación completada: {len(res)} puntos de operación resueltos. "
                    "Los resultados están en la pestaña 4."
                )


# ===========================================================================
# TAB 4 — RESULTADOS
# ===========================================================================
with tab4:
    res = st.session_state["results"]
    module = st.session_state["module"]

    if res is None or module is None:
        st.info("Execute uma simulação na aba 3 para visualizar os resultados.")
    else:
        st.subheader("Período visualizado")
        res_view, results_period_label = select_date_window(res, "results_view")

        if res_view.empty:
            st.warning("O intervalo selecionado não contém resultados.")
        else:
            kpi = compute_kpis(res_view, module, infer_timestep_hours(res_view))
            kpi_full = compute_kpis(res, module, infer_timestep_hours(res))
            dt_h = kpi["dt_hours"]
            st.caption(
                f"Indicadores e gráficos calculados para **{results_period_label}** · "
                f"{len(res_view):,} passos · {kpi['duration_h']:.1f} h"
            )

            # ------------------------- KPIs ----------------------------------
            st.subheader("Indicadores do sistema")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Energia no período", f"{kpi['E_period_kWh']:.3f} kWh")
            k2.metric("Média diária", f"{kpi['E_day_avg_kWh']:.3f} kWh/dia")
            k3.metric("Performance Ratio", f"{kpi['PR']:.3f}")
            k4.metric("Fator de capacidade", f"{kpi['CF']*100:.2f} %")

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Potência máxima", f"{kpi['P_max_W']:.1f} W",
                      help="Saída no MPP resolvida pelo SDM.")
            k6.metric(
                "Instante do pico",
                kpi["t_peak"].strftime("%Y-%m-%d %H:%M") if kpi["t_peak"] is not None else "—",
            )
            k7.metric("Eficiência média", f"{kpi['eta_mean']*100:.2f} %",
                      delta=f"{(kpi['eta_mean']-kpi['eta_stc'])*100:+.2f} pp vs STC")
            k8.metric("Potência média", f"{kpi['P_mean_W']:.1f} W")

            k9, k10, k11, k12 = st.columns(4)
            k9.metric("Yield específico", f"{kpi['specific_yield_kWh_kWp']:.2f} kWh/kWp")
            k10.metric("Irradiação no período", f"{kpi['H_period_kWh_m2']:.2f} kWh/m²")
            k11.metric("Energia solar incidente", f"{kpi['E_solar_kWh']:.2f} kWh")
            k12.metric("Tc máxima", f"{kpi['Tc_max_C']:.1f} °C",
                       delta=f"{kpi['Tc_mean_sun_C']:.1f} °C média com sol")

            n_series_v = kpi.get("n_series", kpi.get("n_modules", 1))
            n_parallel_v = kpi.get("n_parallel", 1)
            area_mod_v = kpi.get("area_module_m2", module.stc.area)

            k13, k14, k15, k16 = st.columns(4)
            k13.metric("Configuração do array", f"{n_series_v} série × {n_parallel_v} paralelo")
            k14.metric("Módulos totais", f"{kpi['n_modules']}")
            k15.metric("Potência nominal", f"{kpi['p_nom_kWp']*1000:.0f} Wp")
            k16.metric("Área total", f"{kpi['area_m2']:.2f} m²",
                       help=f"{kpi['n_modules']} × {area_mod_v:.3f} m² por módulo")

            st.divider()

            # ------------------------- Séries temporais ----------------------
            st.subheader("Séries temporais")
            st.plotly_chart(plot_irradiance(res_view), width="stretch", key="chart_irr_results")
            st.plotly_chart(plot_temperatures(res_view), width="stretch", key="chart_temp_results")

            show_ref = st.checkbox(
                "Sobrepor referência linear P = Pnom·G/1000 (somente comparação)", value=False,
                help="O simulador não usa esta expressão; ela aparece apenas como referência.",
            )
            st.plotly_chart(
                plot_power(res_view, show_linear_ref=show_ref),
                width="stretch", key="chart_power",
            )

            if show_ref and kpi["E_period_kWh"] > 0:
                err = (kpi["E_lineal_ref_kWh"] - kpi["E_period_kWh"]) / kpi["E_period_kWh"] * 100
                st.caption(
                    f"A aproximação linear prevê {kpi['E_lineal_ref_kWh']:.3f} kWh frente a "
                    f"{kpi['E_period_kWh']:.3f} kWh do SDM: diferença de **{err:+.1f} %**."
                )

            st.plotly_chart(
                plot_efficiency(res_view, module.stc.efficiency_stc),
                width="stretch", key="chart_efficiency",
            )
            st.plotly_chart(plot_energy(res_view, dt_h), width="stretch", key="chart_energy")

            st.divider()

            # ------------------------- Curvas I-V / P-V -----------------------
            st.subheader("Curvas I-V e P-V nas condições selecionadas")

            peak_ts = kpi["t_peak"] if kpi["t_peak"] is not None else res_view.index[0]
            available_dates = sorted(set(res_view.index.date))
            default_date_index = available_dates.index(peak_ts.date()) if peak_ts.date() in available_dates else 0

            cv1, cv2, cv3 = st.columns([1, 1, 1.2])
            with cv1:
                curve_date = st.selectbox(
                    "Data", available_dates, index=default_date_index,
                    format_func=lambda d: d.isoformat(), key="curve_date",
                )
                day_rows = res_view[res_view.index.date == curve_date]
                time_labels = day_rows.index.strftime("%H:%M:%S").tolist()
                peak_label = peak_ts.strftime("%H:%M:%S") if peak_ts.date() == curve_date else time_labels[0]
                if peak_label not in time_labels:
                    peak_label = time_labels[0]
                sel_time = st.select_slider(
                    "Horário", options=time_labels, value=peak_label, key="curve_time",
                )

            selected_ts = day_rows.index[day_rows.index.strftime("%H:%M:%S") == sel_time][0]
            row = day_rows.loc[selected_ts]
            with cv2:
                st.metric("G / Tc", f"{row['G']:.0f} W/m² · {row['Tc']:.1f} °C")
            with cv3:
                st.metric(
                    "MPP resolvido",
                    f"{row['Pmp']:.1f} W · {row['Vmp']:.1f} V · {row['Imp']:.2f} A",
                )

            curves = []
            if row["G"] > SOLVER["g_min"]:
                p_op = translate_params(module.sdm, module.stc, row["G_eff"], row["Tc"])
                V, I, P = iv_curve(p_op)
                curves.append({
                    "label": f"{selected_ts:%Y-%m-%d %H:%M} · {row['G']:.0f} W/m², {row['Tc']:.0f} °C",
                    "V": V, "I": I, "P": P,
                    "mpp": {"Vmp": row["Vmp"], "Imp": row["Imp"], "Pmp": row["Pmp"]},
                })
            p_stc = translate_params(module.sdm, module.stc, G_REF, T_REF_C)
            Vs, Is, Ps = iv_curve(p_stc)
            curves.append({
                "label": "STC (referência)", "V": Vs, "I": Is, "P": Ps,
                "mpp": find_mpp(p_stc), "color": "#9AA5B1",
            })
            st.plotly_chart(plot_iv_pv(curves), width="stretch", key="chart_iv_pv_instant")

            with st.expander("Famílias de curvas I-V (validação física do modelo)"):
                fam1, fam2 = st.columns(2)
                with fam1:
                    fam_g = {}
                    for g in (200, 400, 600, 800, 1000):
                        p = translate_params(module.sdm, module.stc, g, 25.0)
                        V, I, _ = iv_curve(p, n_points=200)
                        fam_g[f"{g} W/m²"] = {"V": V, "I": I, "mpp": find_mpp(p)}
                    st.plotly_chart(
                        plot_iv_family(fam_g, "Tensão V [V]",
                                       "Efeito da irradiância (Tc = 25 °C)"),
                        width="stretch", key="chart_family_irr",
                    )
                with fam2:
                    fam_t = {}
                    for t in (15, 25, 45, 65):
                        p = translate_params(module.sdm, module.stc, G_REF, float(t))
                        V, I, _ = iv_curve(p, n_points=200)
                        fam_t[f"{t} °C"] = {"V": V, "I": I, "mpp": find_mpp(p)}
                    st.plotly_chart(
                        plot_iv_family(fam_t, "Tensão V [V]",
                                       "Efeito da temperatura (G = 1000 W/m²)"),
                        width="stretch", key="chart_family_temp",
                    )

            st.plotly_chart(plot_mpp_trajectory(res_view), width="stretch", key="chart_mpp_traj")

            st.divider()

            # ------------------------- Downloads ------------------------------
            st.subheader("Exportação")
            st.caption("Os arquivos abaixo correspondem à simulação completa, não apenas ao recorte exibido.")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ Série temporal simulada (CSV)",
                    res.to_csv().encode("utf-8"),
                    file_name="pv_simulacao_series.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with d2:
                kpi_out = {
                    k: (v.strftime("%Y-%m-%d %H:%M") if hasattr(v, "strftime") else v)
                    for k, v in kpi_full.items()
                }
                st.download_button(
                    "⬇️ KPIs da simulação completa (CSV)",
                    pd.Series(kpi_out, name="valor").to_csv().encode("utf-8"),
                    file_name="pv_simulacao_kpis.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with st.expander("Tabela de resultados do intervalo exibido (primeiras 200 linhas)"):
                st.dataframe(res_view.head(200), width="stretch")


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
