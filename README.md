# ☀️ PV Simulator — Single Diode Model

Simulador de ingeniería para módulos fotovoltaicos que resuelve la **ecuación completa del
circuito equivalente de un diodo** en cada instante de tiempo. **No** se usa en ningún punto
la aproximación `P = Pnom · G/1000`: la potencia sale siempre de localizar numéricamente el
máximo de la curva P-V obtenida del modelo físico.

---

## 1. Modelo matemático

### 1.1 Ecuación del SDM

$$
I = I_{ph} - I_0\left[\exp\!\left(\frac{V + I R_s}{n N_s V_t}\right) - 1\right] - \frac{V + I R_s}{R_{sh}}
$$

| Símbolo | Significado | Unidad |
|---|---|---|
| `Iph` (`IL`) | Corriente fotogenerada | A |
| `I0` | Corriente de saturación inversa del diodo | A |
| `Rs` | Resistencia serie | Ω |
| `Rsh` | Resistencia paralelo (shunt) | Ω |
| `n` | Factor de idealidad | – |
| `Ns` | Nº de células en serie | – |
| `Vt = kT/q` | Tensión térmica de una célula | V |
| `a = n·Ns·Vt` | Tensión térmica modificada del módulo | V |

La ecuación es **implícita** en `I`. Se resuelve por tres caminos independientes que se
verifican cruzadamente (coinciden a ~1e-12 A):

1. **Función W de Lambert** — solución exacta en forma cerrada. Es la usada en la simulación
   temporal (rápida y vectorizable). Se evalúa `W` a partir de `ln(z)` con expansión
   asintótica para evitar overflow.
2. **Brent (`scipy.optimize.brentq`)** — `f(I)=0` es monótona decreciente, el bracketing
   siempre es válido. Máxima robustez.
3. **Newton-Raphson** con derivada analítica, con *fallback* automático a Brent.

### 1.2 Traslación a condiciones de operación — De Soto et al. (2006)

$$
a = a_{ref}\frac{T_c}{T_{ref}} \qquad
I_L = \frac{G}{G_{ref}}\left[I_{L,ref} + \alpha_{Isc}(T_c - T_{ref})\right]
$$

$$
I_0 = I_{0,ref}\left(\frac{T_c}{T_{ref}}\right)^{3}\exp\left[\frac{E_{g,ref}}{kT_{ref}} - \frac{E_g}{kT_c}\right]
\qquad R_{sh} = R_{sh,ref}\frac{G_{ref}}{G} \qquad R_s = \text{cte}
$$

con `Eg = Eg_ref · [1 + dEg/dT · (Tc − Tref)]` (silicio: `Eg_ref = 1.121 eV`).

### 1.3 Modelo térmico (NOCT, editable)

$$
T_c = T_{amb} + \frac{NOCT - 20}{800}\,G
$$

Se incluye también el modelo de **Sandia** como alternativa (`models/temperature_model.py`).

### 1.4 Extracción de los 5 parámetros

Los fabricantes **no publican** `IL, I0, Rs, Rsh, n`. Se estiman resolviendo el sistema no
lineal de 5 ecuaciones sobre los datos de catálogo:

| Ec. | Condición |
|---|---|
| E1 | `I(V=0) = Isc` |
| E2 | `I(V=Voc) = 0` |
| E3 | `I(Vmp) = Imp` |
| E4 | `dP/dV = 0` en el MPP |
| E5 | `dVoc/dT = β_Voc` (catálogo), impuesta numéricamente |

Resolución: `scipy.optimize.least_squares` (Trust-Region-Reflective) sobre el vector escalado
`[IL, log₁₀(I0), Rs, log₁₀(Rsh), n]` con cotas físicas. El escalado logarítmico es
imprescindible: `I0 ~ 1e-11 A` y `Rsh ~ 1e3 Ω` difieren en 14 órdenes de magnitud.

> El coeficiente `γ_Pmax` **no** se impone en el ajuste → sirve como validación cruzada
> independiente (error obtenido: 3.8 % – 7.3 % frente al catálogo).

### 1.5 Búsqueda del MPP

En cada instante: barrido grueso vectorizado de `P(V)` sobre `[0, Voc]` + refinamiento con
minimización acotada de `−P(V)` (Brent, `minimize_scalar(method="bounded")`).
**Hipótesis del sistema:** MPPT ideal ⇒ `Pout(t) = Pmp(t)`. No se implementa P&O ni IncCond.

---

## 2. Instalación

```bash
git clone <repo>            # o descomprimir el .zip
cd PV_Simulator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Abre en `http://localhost:8501`.

### Streamlit Cloud
Subir el repo a GitHub → *New app* → archivo principal `app.py`. El `requirements.txt` es
suficiente (todas las dependencias son *wheels* puras).

### Validación
```bash
python -m tests.validate
python -m unittest tests.test_profile_loader -v
```

---

## 3. Arquitectura

```
PV_Simulator/
├── app.py                       # Interfaz Streamlit (4 pestañas). Sin física.
├── models/
│   ├── single_diode.py          # Ecuación del SDM + solvers (LambertW/Brent/Newton)
│   ├── pv_module.py             # Dataclasses: ModuleSTC, SDMParams, SDMOperating, PVModule
│   ├── temperature_model.py     # Tc = f(Tamb, G): NOCT y Sandia
│   └── irradiance_model.py      # Perfiles sintéticos + detección/completado de CSV
├── config/
│   ├── settings.py              # Constantes físicas, STC, bounds, estilo. CERO magic numbers.
│   └── pv_database.py           # Base de módulos (5 × Canadian Solar poly)
├── simulation/
│   ├── solver.py                # Extracción de los 5 parámetros + traslación De Soto
│   ├── mpp.py                   # Búsqueda del MPP + motor de simulación temporal
│   └── energy.py                # Integración energética + KPIs
├── visualization/
│   └── plots.py                 # Plotly (tema claro). Sin cálculo físico.
├── data/custom_profiles/        # CSV de ejemplo (1 min)
├── tests/validate.py            # Validación física y numérica
├── tests/test_profile_loader.py # Pruebas del cargador y completado temporal
├── requirements.txt
└── README.md
```

Separación estricta: **física** (`models/`) · **parámetros** (`config/`) ·
**resolución** (`simulation/`) · **gráficos** (`visualization/`) · **interfaz** (`app.py`).

---

## 4. Base de datos de módulos

Cinco módulos **policristalinos Canadian Solar** de distintas generaciones y potencias
(valores STC de catálogo público):

| Modelo | Pnom | Voc | Isc | Vmp | Imp | Ns | α_Isc | β_Voc | γ_Pmax | NOCT |
|---|---|---|---|---|---|---|---|---|---|---|
| CS6P-250P (Quartech) | 250 W | 37.2 V | 8.87 A | 30.1 V | 8.30 A | 60 | +0.065 %/°C | −0.34 %/°C | −0.43 %/°C | 45 °C |
| CS6P-260P (Quartech) | 260 W | 37.5 V | 9.10 A | 30.4 V | 8.56 A | 60 | +0.065 | −0.34 | −0.43 | 45 °C |
| CS6K-275P (KuPower) | 275 W | 38.0 V | 9.45 A | 31.0 V | 8.88 A | 60 | +0.053 | −0.31 | −0.41 | 45 °C |
| CS6X-315P (MaxPower) | 315 W | 45.1 V | 9.18 A | 36.6 V | 8.61 A | 72 | +0.065 | −0.34 | −0.43 | 45 °C |
| CS6U-330P (MaxPower2) | 330 W | 45.6 V | 9.45 A | 37.2 V | 8.88 A | 72 | +0.053 | −0.31 | −0.41 | 45 °C |

**Añadir un módulo nuevo:** agregar una entrada al diccionario `MODULE_DB` en
`config/pv_database.py`. Nada más. Si se dispone de los parámetros SDM (p. ej. de la base
CEC/SAM o de un ensayo flash), rellenar el campo `"sdm"`; si queda en `None` se estiman
automáticamente. Todo es editable desde la pestaña 1.

---

## 5. Uso de la interfaz

| Pestaña | Contenido |
|---|---|
| **1 · Configuración del panel** | Selector fabricante/modelo (+ *Personalizado*). Parámetros eléctricos STC y coeficientes de temperatura editables. Botón de estimación de los 5 parámetros SDM, residuos del ajuste, y verificación catálogo-vs-modelo con la curva I-V/P-V en STC. |
| **2 · Irradiancia y temperatura** | **Modo A:** carga de CSV con detección de `Timestamp`, `G`, `GHI_REAL`, `GHI_PREDITO` y temperatura. Permite elegir la serie de GHI, completar las noches con `G=0`, identificar lagunas diurnas, generar temperatura día/noche o por curva estándar y filtrar gráficos por día o intervalo. **Modo B:** generador de perfiles sintéticos. |
| **3 · Modelo y simulación** | Resumen del módulo, de los parámetros SDM y de las condiciones. Configuración del generador (serie × paralelo, suciedad). Botón **SIMULAR** con barra de progreso. |
| **4 · Resultados** | KPIs + 8 gráficos + curvas I-V/P-V en el instante seleccionado + familias de curvas paramétricas + trayectoria del MPP + exportación a CSV. |


### CSV de previsão (Vitor → modelo solar)

O carregador aceita cabeçalhos flexíveis. Para o fluxo atual do projeto, um arquivo como o abaixo é reconhecido automaticamente:

```csv
Timestamp,GHI_REAL,GHI_PREDITO
2016-09-01 06:45:00,4.14,7.58
2016-09-01 06:46:00,4.98,7.77
```

Quando existem `GHI_REAL` e `GHI_PREDITO`, a interface seleciona `GHI_PREDITO` por padrão, mas permite trocar a coluna. Se não houver temperatura, a interface oferece:

1. temperatura constante de dia e de noite; ou
2. curva térmica padrão com temperatura mínima e máxima.

Ao completar o eixo temporal, a plataforma adiciona as colunas de rastreabilidade:

| Coluna | Significado |
|---|---|
| `is_original` | registro existente no CSV recebido |
| `is_filled` | registro criado pela plataforma |
| `fill_type` | `original`, `preenchido_noite` ou `preenchido_lacuna_diurna` |
| `Tamb_filled` | temperatura gerada ou interpolada pela plataforma |

As lacunas diurnas também recebem `G=0`, mas são destacadas em vermelho no mapa de cobertura para não serem confundidas com noite.

### KPIs calculados

| KPI | Definición | Unidad |
|---|---|---|
| Energía no período | `Σ P(t)·Δt / 1000` | kWh |
| Média diária | `E_período / duração[dias]` | kWh/dia |
| Yield específico `Yf` | `E_período / Pnom[kW]` | kWh/kWp |
| Performance Ratio `PR` | `Yf / H_período`, com `H = Σ G·Δt / 1000` | – |
| Fator de capacidade `CF` | `E_período / (Pnom[kW] · duração[h])` | – |
| Potência máxima + instante do pico | `max P(t)` | W, data-hora |
| Eficiência média | `E_período / (H_período · Área)` | – |
| Potência média | `mean P(t)` | W |

---

## 6. Validación (`python -m tests.validate`)

Se ejecuta sobre **los 5 módulos** de la base:

| # | Comprobación | Resultado |
|---|---|---|
| V1 | La extracción reproduce el catálogo (Isc, Voc, Vmp, Imp, Pmax) | error < 0.11 % |
| V2 | Lambert W ≡ Brent ≡ Newton | dif. máx. ~1e-12 A |
| V3 | **Voc disminuye con la temperatura**; `β_Voc` del modelo ≈ catálogo | error < 0.6 % |
| V4 | **Isc y Pmp aumentan con la irradiancia** (monotonía estricta) | ✔ |
| V5 | **El MPP es un punto interior** de la curva P-V (`0 < Vmp < Voc`) | `Vmp/Voc ≈ 0.81`, `FF ≈ 0.76` |
| V6 | `γ_Pmax` del modelo ≈ catálogo (*no impuesto en el ajuste*) | error 3.8 – 7.3 % |
| V7 | **Energía diaria con unidades y órdenes correctos** | 2.06 kWh para un CS6U-330P, día soleado de verano; PR = 0.89; η media 15.1 % < η STC 16.97 % |

Rendimiento: **1440 pasos (1 día a 1 min) en ≈ 0.5 s**.

---

## 7. Extensión futura (arquitectura preparada)

* **Nuevos fabricantes** → nuevas entradas en `MODULE_DB` (cero cambios de código).
* **Sistemas completos** → `simulate_timeseries()` ya acepta `n_series` / `n_parallel` y
  pérdidas; el siguiente paso natural es un módulo `models/inverter.py` (curva de eficiencia,
  clipping DC/AC, ventana de MPPT).
* **Baterías / EMS** → la salida `results["P_array"]` es una serie de potencia lista para
  alimentar un despacho: crear `simulation/dispatch.py` y `models/battery.py`.
* **Modelos energéticos** → los KPIs y la serie horaria se exportan a CSV y pueden acoplarse
  a un modelo de expansión o a un balance de sistema.

---

## 8. Referencias

* De Soto, W., Klein, S.A., Beckman, W.A. (2006). *Improvement and validation of a model for
  photovoltaic array performance*. Solar Energy 80(1), 78-88.
* Villalva, M.G., Gazoli, J.R., Filho, E.R. (2009). *Comprehensive approach to modeling and
  simulation of photovoltaic arrays*. IEEE Trans. Power Electronics 24(5).
* Jain, A., Kapoor, A. (2004). *Exact analytical solutions of the parameters of real solar
  cells using Lambert W-function*. Solar Energy Materials & Solar Cells 81(2).
* King, D.L. et al. (2004). *Photovoltaic Array Performance Model*. Sandia SAND2004-3535.
* IEC 61724 — *Photovoltaic system performance monitoring*.
