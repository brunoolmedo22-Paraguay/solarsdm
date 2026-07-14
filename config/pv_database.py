"""
config/pv_database.py
=====================
Base de datos de módulos fotovoltaicos.

Fuente: hojas de datos públicas de Canadian Solar (valores en STC:
1000 W/m2, AM 1.5, Tcell = 25 °C). Se incluyen cinco módulos
POLICRISTALINOS de distintas generaciones y potencias:

    CS6P-250P  (Quartech,  60 células, 2013-2015)
    CS6P-260P  (Quartech,  60 células, 2013-2015)
    CS6K-275P  (KuPower,   60 células, 2016-2018)
    CS6X-315P  (MaxPower,  72 células, 2014-2016)
    CS6U-330P  (MaxPower2, 72 células, 2016-2019)

Los parámetros del SDM (IL, I0, Rs, Rsh, n) NO son publicados por el
fabricante. El campo `sdm` queda por tanto en None y se ESTIMA en tiempo de
ejecución con `simulation.solver.extract_sdm_params()`. Si el usuario dispone
de mejores parámetros (p. ej. de la base CEC o de un ensayo flash), puede:

    * cargarlos aquí, rellenando el diccionario `sdm`;
    * o editarlos directamente desde la pestaña 1 de la aplicación.

--- CÓMO AÑADIR UN MÓDULO NUEVO ---------------------------------------------
Basta con agregar una entrada al diccionario MODULE_DB con la misma estructura.
No hay que tocar ninguna otra parte del código.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional

from models.pv_module import ModuleSTC, SDMParams, PVModule

CUSTOM_KEY = "Personalizado"

# ---------------------------------------------------------------------------
# Datos de catálogo (STC)
# ---------------------------------------------------------------------------
MODULE_DB: Dict[str, dict] = {
    "CS6P-250P": {
        "stc": dict(
            manufacturer="Canadian Solar",
            model="CS6P-250P (Quartech)",
            technology="Poly-Si",
            p_nom=250.0, v_oc=37.2, i_sc=8.87, v_mp=30.1, i_mp=8.30,
            n_cells=60,
            alpha_isc_pct=0.065, beta_voc_pct=-0.34, gamma_pmax_pct=-0.43,
            noct=45.0, length=1.638, width=0.982,
            notes="4 busbars. Eficiencia de catálogo 15.54 %.",
        ),
        "sdm": None,
    },
    "CS6P-260P": {
        "stc": dict(
            manufacturer="Canadian Solar",
            model="CS6P-260P (Quartech)",
            technology="Poly-Si",
            p_nom=260.0, v_oc=37.5, i_sc=9.10, v_mp=30.4, i_mp=8.56,
            n_cells=60,
            alpha_isc_pct=0.065, beta_voc_pct=-0.34, gamma_pmax_pct=-0.43,
            noct=45.0, length=1.638, width=0.982,
            notes="Misma plataforma que el CS6P-250P. Eficiencia 16.16 %.",
        ),
        "sdm": None,
    },
    "CS6K-275P": {
        "stc": dict(
            manufacturer="Canadian Solar",
            model="CS6K-275P (KuPower)",
            technology="Poly-Si",
            p_nom=275.0, v_oc=38.0, i_sc=9.45, v_mp=31.0, i_mp=8.88,
            n_cells=60,
            alpha_isc_pct=0.053, beta_voc_pct=-0.31, gamma_pmax_pct=-0.41,
            noct=45.0, length=1.650, width=0.992,
            notes="Generación KuPower, 5 busbars. Eficiencia 16.80 %.",
        ),
        "sdm": None,
    },
    "CS6X-315P": {
        "stc": dict(
            manufacturer="Canadian Solar",
            model="CS6X-315P (MaxPower)",
            technology="Poly-Si",
            p_nom=315.0, v_oc=45.1, i_sc=9.18, v_mp=36.6, i_mp=8.61,
            n_cells=72,
            alpha_isc_pct=0.065, beta_voc_pct=-0.34, gamma_pmax_pct=-0.43,
            noct=45.0, length=1.954, width=0.982,
            notes="72 células. Módulo típico de planta utility. Eficiencia 16.42 %.",
        ),
        "sdm": None,
    },
    "CS6U-330P": {
        "stc": dict(
            manufacturer="Canadian Solar",
            model="CS6U-330P (MaxPower2)",
            technology="Poly-Si",
            p_nom=330.0, v_oc=45.6, i_sc=9.45, v_mp=37.2, i_mp=8.88,
            n_cells=72,
            alpha_isc_pct=0.053, beta_voc_pct=-0.31, gamma_pmax_pct=-0.41,
            noct=45.0, length=1.960, width=0.992,
            notes="Eficiencia 16.97 %. Muy usado en plantas en Brasil/Paraguay.",
        ),
        "sdm": None,
    },
}


# ---------------------------------------------------------------------------
# API de la base de datos
# ---------------------------------------------------------------------------
def list_manufacturers() -> List[str]:
    """Fabricantes disponibles (+ opción personalizada)."""
    mans = sorted({e["stc"]["manufacturer"] for e in MODULE_DB.values()})
    return mans + [CUSTOM_KEY]


def list_models(manufacturer: str) -> List[str]:
    """Modelos de un fabricante dado."""
    if manufacturer == CUSTOM_KEY:
        return [CUSTOM_KEY]
    return [k for k, e in MODULE_DB.items() if e["stc"]["manufacturer"] == manufacturer]


def get_module(key: str) -> PVModule:
    """Devuelve un PVModule (con sdm=None si no hay parámetros publicados)."""
    if key not in MODULE_DB:
        raise KeyError(f"Módulo no encontrado en la base de datos: {key!r}")
    entry = MODULE_DB[key]
    stc = ModuleSTC(**entry["stc"])

    sdm: Optional[SDMParams] = None
    if entry.get("sdm"):
        sdm = SDMParams(n_cells=stc.n_cells, source="datasheet", **entry["sdm"])

    return PVModule(stc=stc, sdm=sdm)


def default_custom_stc() -> ModuleSTC:
    """Plantilla editable para el modo 'Personalizado'."""
    return ModuleSTC(
        manufacturer="Personalizado",
        model="Módulo definido por el usuario",
        technology="Poly-Si",
        p_nom=300.0, v_oc=40.0, i_sc=9.50, v_mp=32.5, i_mp=9.23,
        n_cells=60,
        alpha_isc_pct=0.05, beta_voc_pct=-0.30, gamma_pmax_pct=-0.39,
        noct=45.0, length=1.700, width=1.000,
        notes="Editable en su totalidad.",
    )
