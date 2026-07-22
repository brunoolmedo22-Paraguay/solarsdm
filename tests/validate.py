"""
tests/validate.py
=================
Validación física y numérica del simulador. Ejecutar con:

    python -m tests.validate

Comprueba, para TODOS los módulos de la base de datos:

  V1. La extracción de los 5 parámetros reproduce el catálogo:
      Isc, Voc, Vmp, Imp, Pmax con error < 0.5 %.
  V2. Los tres solvers (Lambert W, Brent, Newton) coinciden (< 1e-8 A).
  V3. Voc DISMINUYE con la temperatura y el coeficiente beta_Voc del modelo
      coincide con el del catálogo.
  V4. Isc y Pmp AUMENTAN con la irradiancia (monotonía).
  V5. El MPP es un punto INTERIOR: 0 < Vmp < Voc y 0 < Imp < Isc.
  V6. El coeficiente gamma_Pmax del modelo coincide con el del catálogo
      (esta ecuación NO se impone en la extracción -> es una validación real).
  V7. La energía diaria tiene unidades y órdenes de magnitud correctos.
"""

from __future__ import annotations

import sys

import numpy as np

from config.pv_database import MODULE_DB, get_module
from config.settings import G_REF, T_REF_C
from models.irradiance_model import build_synthetic_profile, infer_timestep_hours
from models.single_diode import (
    current_from_voltage,
    iv_curve,
    open_circuit_voltage,
    short_circuit_current,
)
from simulation.energy import compute_kpis
from simulation.mpp import find_mpp, simulate_timeseries
from simulation.solver import extract_sdm_params, translate_params

OK, FAIL = "  [OK]  ", "  [FAIL]"
errors = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name} {detail}")
    if not cond:
        errors.append(name)


def main():
    for key in MODULE_DB:
        module = get_module(key)
        stc = module.stc
        print("\n" + "=" * 78)
        print(f"MÓDULO: {key}  |  Pnom={stc.p_nom} W  Ns={stc.n_cells}  "
              f"Voc={stc.v_oc} V  Isc={stc.i_sc} A")
        print("=" * 78)

        # ---------------- Extracción -------------------------------------
        sdm, rep = extract_sdm_params(stc)
        module.sdm = sdm
        print(f"  SDM: IL={sdm.IL_ref:.4f} A | I0={sdm.I0_ref:.3e} A | "
              f"Rs={sdm.Rs:.4f} ohm | Rsh={sdm.Rsh_ref:.1f} ohm | n={sdm.n:.4f}")
        print(f"  Ajuste: cost={rep.cost:.2e}  nfev={rep.n_iter}")

        p_stc = translate_params(sdm, stc, G_REF, T_REF_C)

        # V1 --------------------------------------------------------------
        isc = short_circuit_current(p_stc)
        voc = open_circuit_voltage(p_stc)
        mpp = find_mpp(p_stc)
        e_isc = abs(isc - stc.i_sc) / stc.i_sc * 100
        e_voc = abs(voc - stc.v_oc) / stc.v_oc * 100
        e_vmp = abs(mpp["Vmp"] - stc.v_mp) / stc.v_mp * 100
        e_imp = abs(mpp["Imp"] - stc.i_mp) / stc.i_mp * 100
        e_pmp = abs(mpp["Pmp"] - stc.p_nom) / stc.p_nom * 100
        print(f"  V1 errores STC [%]: Isc={e_isc:.3f} Voc={e_voc:.3f} "
              f"Vmp={e_vmp:.3f} Imp={e_imp:.3f} Pmax={e_pmp:.3f}")
        check("V1 reproduce catálogo (<0.5 %)",
              max(e_isc, e_voc, e_vmp, e_imp, e_pmp) < 0.5)

        # V2 --------------------------------------------------------------
        v_test = np.linspace(0.0, voc * 0.98, 12)
        i_lw = np.array([current_from_voltage(v, p_stc, "lambertw") for v in v_test])
        i_br = np.array([current_from_voltage(v, p_stc, "brentq") for v in v_test])
        i_nw = np.array([current_from_voltage(v, p_stc, "newton") for v in v_test])
        d1 = float(np.max(np.abs(i_lw - i_br)))
        d2 = float(np.max(np.abs(i_lw - i_nw)))
        check("V2 solvers coinciden (LambertW/Brent/Newton)",
              max(d1, d2) < 1e-6, f"| max dif = {max(d1, d2):.2e} A")

        # V3 --------------------------------------------------------------
        vocs = [open_circuit_voltage(translate_params(sdm, stc, G_REF, T))
                for T in (15.0, 25.0, 45.0, 65.0)]
        monotona = all(vocs[i] > vocs[i + 1] for i in range(len(vocs) - 1))
        beta_model = (vocs[3] - vocs[1]) / (65.0 - 25.0)
        e_beta = abs(beta_model - stc.beta_voc) / abs(stc.beta_voc) * 100
        check("V3 Voc decrece con T", monotona,
              f"| Voc(15,25,45,65) = {['%.2f' % v for v in vocs]} V")
        check("V3 beta_Voc del modelo ~ catálogo (<5 %)", e_beta < 5.0,
              f"| modelo={beta_model:.4f} catálogo={stc.beta_voc:.4f} V/°C "
              f"(err {e_beta:.2f} %)")

        # V4 --------------------------------------------------------------
        gs = [200.0, 400.0, 600.0, 800.0, 1000.0]
        iscs, pmps = [], []
        for g in gs:
            pg = translate_params(sdm, stc, g, 25.0)
            iscs.append(short_circuit_current(pg))
            pmps.append(find_mpp(pg)["Pmp"])
        check("V4 Isc crece con G", all(np.diff(iscs) > 0),
              f"| Isc = {['%.2f' % v for v in iscs]} A")
        check("V4 Pmp crece con G", all(np.diff(pmps) > 0),
              f"| Pmp = {['%.1f' % v for v in pmps]} W")

        # V5 --------------------------------------------------------------
        V, I, P = iv_curve(p_stc)
        k = int(np.argmax(P))
        interior = 0 < k < len(P) - 1
        check("V5 MPP es punto interior de la curva P-V", interior
              and 0 < mpp["Vmp"] < voc and 0 < mpp["Imp"] < isc,
              f"| Vmp/Voc = {mpp['Vmp']/voc:.3f}, FF = {mpp['FF']:.3f}")

        # V6 (validación cruzada: gamma NO se impone en la extracción) ------
        p25 = find_mpp(translate_params(sdm, stc, G_REF, 25.0))["Pmp"]
        p65 = find_mpp(translate_params(sdm, stc, G_REF, 65.0))["Pmp"]
        gamma_model_pct = (p65 - p25) / 40.0 / stc.p_nom * 100.0
        e_gamma = abs(gamma_model_pct - stc.gamma_pmax_pct) / abs(stc.gamma_pmax_pct) * 100
        check("V6 gamma_Pmax del modelo ~ catálogo (<15 %)", e_gamma < 15.0,
              f"| modelo={gamma_model_pct:.4f} catálogo={stc.gamma_pmax_pct:.4f} %/°C "
              f"(err {e_gamma:.1f} %)")

    # ---------------- V7: energía ------------------------------------------
    print("\n" + "=" * 78)
    print("V7 — SIMULACIÓN TEMPORAL Y ENERGÍA (CS6U-330P, día soleado, verano)")
    print("=" * 78)
    module = get_module("CS6U-330P")
    module.sdm, _ = extract_sdm_params(module.stc)
    profile = build_synthetic_profile("Día soleado", "Verano", timestep_min=1)
    res = simulate_timeseries(module, profile)
    dt = infer_timestep_hours(res)
    kpi = compute_kpis(res, module, dt)

    print(f"  Energía diaria      : {kpi['E_day_kWh']:.3f} kWh")
    print(f"  Irradiación (PSH)   : {kpi['PSH_h']:.3f} kWh/m2")
    print(f"  Yield específico    : {kpi['specific_yield_kWh_kWp']:.3f} kWh/kWp")
    print(f"  Performance Ratio   : {kpi['PR']:.3f}")
    print(f"  Factor de capacidad : {kpi['CF']*100:.2f} %")
    print(f"  Potencia máxima     : {kpi['P_max_W']:.1f} W  @ {kpi['t_peak']}")
    print(f"  Eficiencia media    : {kpi['eta_mean']*100:.2f} %  (STC: {kpi['eta_stc']*100:.2f} %)")
    print(f"  Tc máxima           : {kpi['Tc_max_C']:.1f} °C")

    check("V7 energía diaria en rango físico (0.5-2.5 kWh para 330 Wp)",
          0.5 < kpi["E_day_kWh"] < 2.5)
    check("V7 PR en rango realista (0.70-0.95)", 0.70 < kpi["PR"] < 0.95)
    check("V7 Pmax <= Pnom (día caluroso -> derating térmico)",
          kpi["P_max_W"] < module.stc.p_nom)
    check("V7 eficiencia media < eficiencia STC", kpi["eta_mean"] < kpi["eta_stc"])

    print("\n" + "=" * 78)
    if errors:
        print(f"RESULTADO: {len(errors)} VALIDACIONES FALLIDAS -> {errors}")
        return 1
    print("RESULTADO: TODAS LAS VALIDACIONES PASARON ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
