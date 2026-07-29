# Inactivos — Scoring de Compromiso

Pipeline de scoring para la población de asociados **inactivos** de Coomeva.
Para cada cédula/periodo en `data/Insumos/Poblacion_Inactivos.xlsx` calcula un
**`score_compromiso`** en escala [0, 100], resultado de combinar 5 dimensiones
de comportamiento, y además genera una **priorización interna** dentro del
segmento de score "Alto" para enfocar la gestión comercial.

Este repo es la implementación productiva/local de un modelo diseñado
originalmente en el trabajo de grado de maestría en ciencia de datos del autor
(proyecto hermano `MCD-PDG1`), adaptado para correr contra datos reales de
BodegaCorporativa.

> **Nota de estado:** el modelo original de la tesis tenía 4 dimensiones (D1
> Esfuerzo, D2 Enganche, D3 Recencia, D4 Vínculo). El código de este repo ya
> implementa **5 dimensiones (D1-D5)** — se agregó D1 Severidad de inactividad
> (mora, recaudo, ofertas de reactivación) y D5 Externo (alertas de
> centrales/hábito de pago externas), con pesos que ya suman 1.0 en
> `configs/config.yml`. Este README documenta el pipeline **tal como está hoy
> en el código**.

---

## Arquitectura

El flujo completo vive orquestado en `main.py` (y, en versión parcial y en
construcción, en `train.py` — ver [Estado actual](#estado-actual-y-limitaciones-conocidas)).
Cada etapa es un módulo independiente en `src/`, se comunica con la siguiente
por archivos en disco (parquet/xlsx), y puede correrse por separado siempre
que el insumo previo ya exista.

```
                 ┌─────────────────────┐
  BodegaCorp. →  │  1. EXTRACCIÓN       │  src/extraccion/extract_raw.py
  Excels red   →  │  extract_raw()       │  → data/raw/*.parquet
                 └──────────┬──────────┘
                            │ dict[str, DataFrame]
                            ▼
                 ┌─────────────────────┐
                 │  2. TRANSFORMACIÓN   │  src/transformacion/build_analytic_score.py
                 │  build_analytic_score│  → data/analytic/analytic_score_base.parquet
                 └──────────┬──────────┘
                            │ 1 fila por Periodo+Id, variables D1-D5 crudas
                            ▼
                 ┌─────────────────────┐
                 │  3. SCORING          │  src/Scoring/build_score.py
                 │  build_score()       │  → data/scoring/scoring_inactivos.parquet
                 └──────────┬──────────┘  → models/score/*.pkl (normalizadores)
                            │ score_compromiso [0,100] + categoria_score
                            ▼
                 ┌─────────────────────┐
                 │  4. CARACTERIZACIÓN  │  src/Scoring/caracterizar_score.py
                 │  caracterizar_score()│  → reports/scoring/*.jpg, *.csv
                 └──────────┬──────────┘  (diagnóstico, no alimenta nada después)
                            │
                            ▼
                 ┌─────────────────────┐
                 │  5. PRIORIZACIÓN     │  src/Scoring/zoom_alta.py
                 │  ALTA (zoom)         │  → data/scoring/priorizacion_alta.xlsx
                 │  build_zoom()        │
                 └─────────────────────┘
```

### Estructura de carpetas

```
Inactivos/
├── main.py                    # entry point: pipeline completo (pasos 1-4)
├── train.py                   # entry point en construcción (ver Estado actual)
├── configs/
│   ├── config.yml             # pesos del score, rutas de insumos, orden CLV
│   └── scoring/cortes_scoring.json   # cortes Bajo/Medio/Alto (se sobreescribe en cada build_score())
├── sql/                        # queries contra BodegaCorporativa (.sql, con placeholders '?'/{ids})
├── src/
│   ├── extraccion/extract_raw.py
│   ├── transformacion/build_analytic_score.py
│   ├── Scoring/
│   │   ├── build_score.py        # normalización + score_compromiso
│   │   ├── caracterizar_score.py # diagnóstico post-hoc
│   │   └── zoom_alta.py          # mini-score de priorización para categoria_score == 'Alto'
│   └── utils/config.py           # load_config() -> lee configs/config.yml
├── data/
│   ├── Insumos/                # Excels fuente (población, CLV, reactivaciones, features, 360)
│   ├── raw/                    # snapshots parquet con fecha de extracción
│   ├── analytic/               # base analítica (variables D1-D5 crudas, sin normalizar)
│   └── scoring/                # score final + priorización de altos
├── models/score/                # MinMaxScaler/OrdinalEncoder persistidos (.pkl), uno por variable
└── reports/scoring/             # histograma, resúmenes por categoría (diagnóstico)
```

---

## Las 5 dimensiones del score

Definidas en `calcular_scoring()` (`src/Scoring/build_score.py`), pesos en
`configs/config.yml → scoring` (ya suman 1.0):

| Dim. | Nombre | Peso | Variables |
|------|--------|------|-----------|
| D1 | Severidad de inactividad | 0.22 | Tiempo inactivo, % mora GECC 6M, recaudo promedio 6M, ofertas de reactivación disponibles, intenciones de retiro último año |
| D2 | Enganche | 0.22 | N° productos, N° empresas del grupo con vínculo, valor perseverancia/ingresos, total eventos/usos, valor capitalizado |
| D3 | Recencia | 0.22 | Distancia a última reactivación, recencia de última mejora de producto, turnos en oficina 12M, PQR último año |
| D4 | Vínculo histórico | 0.22 | Cuotas pagadas vs antigüedad, reactivaciones previas, rango CLV, saldo de aportes, cercanía a perseverancia |
| D5 | Externo | 0.12 | 3 alertas binarias de centrales/hábito de pago externo (invertidas: sin alertas = mejor) |

Cada dimensión es el **promedio simple** (ignorando NaN) de sus variables ya
normalizadas a [0,1]; el score final es la combinación lineal ponderada,
escalada a [0,100]. Luego `categorizar_score()` corta la distribución en
Bajo/Medio/Alto usando `media ± 0.5·std` del lote corrido.

### Normalización

Cada variable pasa por una de estas funciones (todas en `build_score.py`),
según su forma de distribución:

- `normalizar_continua` — MinMax con winsorización a percentiles [5,95].
- `normalizar_log` — igual, pero con `log1p` previo (variables sesgadas: valor
  capitalizado, saldo, distancia de reactivación, etc.).
- `normalizar_zero_inflated` — para variables con masa de ceros estructural
  (ceros → 0.0 directo; positivos → `(0.1, 1.0]` tras log+MinMax).
- `normalizar_categoricas` — `OrdinalEncoder` + MinMax, usada solo para `Clv`.

Cada llamada persiste su(s) artefacto(s) en `models/score/` (un `.pkl` por
variable).

---

## Decisiones clave (no obvias desde el código)

- **Rezago de 1 mes en el periodo de consulta**: `Numcantidadproductos`,
  `v_360` y `demografica` se consultan en el mes *anterior* al periodo
  objetivo de cada cédula (ej. población en 202606 → se consulta 202605).
  `Distancia_ultima_reactivacion` y `Cantidad_Reactivaciones_Previas` sí usan
  el periodo objetivo tal cual, porque comparan contra el Excel maestro de
  reactivaciones, no contra BodegaCorporativa.
- **Reactivaciones históricas** se leen del Excel maestro
  (`config.yml → paths.path_reac`, ruta de red), filtrado a las cédulas de
  `Poblacion_Inactivos` con periodo anterior al objetivo. Cédulas sin
  historial previo no son un error: caen en `Cantidad_Reactivaciones_Previas=0`
  y `Distancia_ultima_reactivacion` usa el fallback `Antiguedad*12`.
- **Vencido vs Cuota en `calcular_promedio_fac_rec`**: `Vencido_Mes` es un
  *stock* acumulado y `Cuota_Mes` un *flujo* mensual — no se suman
  directamente. `Pct_Mora` se calcula como
  `Vencido_Promedio / (Vencido_Promedio + Factura_Total_Promedio)` para evitar
  doble conteo y para que la severidad sea proporcional, no en pesos absolutos.
- **CLV.xlsx** es una copia del mismo archivo usado en `MCD-PDG1`, no
  específico de esta población — cédulas sin cruce quedan con `Clv=NaN`, que
  el encoder trata como categoría desconocida (score 0 en esa variable).
- **`fit_transform`, no `transform`**: todas las funciones `normalizar_*`
  reajustan sus umbrales en cada corrida de `build_score()` y sobrescriben los
  `.pkl` en `models/score/`. Esto significa que **hoy no hay separación
  train/inferencia a nivel del score principal** — cada corrida es, en
  efecto, un reentrenamiento. Ver más abajo.
- **`zoom_alta.py` sí implementa el patrón de inferencia correcto**: a
  diferencia de `build_score.py`, `inferencia_continua()` en `zoom_alta.py`
  **carga** el `.pkl` ya persistido en `models/score/` y solo llama
  `.transform()` — nunca reajusta. Es el único punto del pipeline que separa
  entrenar de inferir, aunque reutiliza (por conveniencia) los artefactos que
  dejó la última corrida de `build_score()`.

---

## Requisitos y setup

- `.env` en la raíz con `CON_BI` (cadena/DSN `pyodbc` a BodegaCorporativa).
- Entorno virtual `.venv313` ya creado en el repo
  (`.venv313\Scripts\python.exe`). Dependencias en `requeriments.txt`
  (pandas, scikit-learn, pyodbc, joblib, pyyaml, python-dotenv, matplotlib,
  pyarrow, openpyxl, entre otras).
- Acceso de red a la ruta UNC configurada en `configs/config.yml → paths.path_reac`
  (Excel maestro de reactivaciones).

## Cómo correr

```powershell
.venv313\Scripts\python.exe main.py
```

Corre extracción → transformación → scoring → caracterización. Genera/actualiza
`data/raw/`, `data/analytic/`, `data/scoring/scoring_inactivos.parquet`,
`models/score/*.pkl` y los reportes de diagnóstico.

La priorización de altos (`build_zoom()`, paso 5) se corre aparte,
actualmente desde `train.py`, y depende de que ya exista
`data/scoring/scoring_inactivos.parquet` (paso 3) y
`data/analytic/analytic_score_base.parquet` (paso 2).

---

## Estado actual y limitaciones conocidas

- **`main.py` y `train.py` están en transición.** Hoy `main.py` corre el
  pipeline completo (extracción → scoring → caracterización) sin generar la
  priorización de altos; `train.py` es una copia parcial, con los pasos 1-4
  comentados, que solo corre `build_zoom()`. Esta separación en dos archivos
  es un trabajo en curso — todavía no hay una separación limpia y explícita
  entre "entrenar" (ajustar los normalizadores una vez) y "puntuar" (aplicar
  esos normalizadores a datos nuevos sin reajustarlos) a nivel de todo el
  pipeline, salvo por el caso ya resuelto en `zoom_alta.py` mencionado arriba.
- **`configs/scoring/cortes_scoring.json` se recalcula en cada corrida** de
  `build_score()` a partir de la media/std del lote corrido — los cortes
  Bajo/Medio/Alto no son fijos entre corridas todavía.