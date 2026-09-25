# Inactivos — Scoring de Compromiso

Pipeline de scoring para la población de asociados **inactivos** de Coomeva.
Para cada cédula calcula un **`score_compromiso`** en escala [0, 100],
resultado de combinar 5 dimensiones de comportamiento, clasifica en
Bajo/Medio/Alto, y genera una **priorización numérica** dentro del segmento
"Alto" para enfocar la gestión comercial.

Este repo es la implementación productiva/local de un modelo diseñado
originalmente en el trabajo de grado de maestría en ciencia de datos del autor
(proyecto hermano `MCD-PDG1`), adaptado para correr contra datos reales de
BodegaCorporativa (SQL Server) y GCC (Oracle).

> **Nota de estado:** el modelo original de la tesis tenía 4 dimensiones (D1
> Esfuerzo, D2 Enganche, D3 Recencia, D4 Vínculo). El código de este repo
> implementa **5 dimensiones (D1-D5)** — se agregó D1 Severidad de inactividad
> (mora, recaudo, intenciones de retiro) y D5 Externo (alertas de
> centrales/hábito de pago externas), con pesos que ya suman 1.0 en
> `configs/config.yml`. Este README documenta el pipeline **tal como está hoy
> en el código**.

---

## Dos entry points: entrenar vs inferir

El pipeline vive en **dos orquestadores separados** en la raíz del repo — no
comparten estado más allá de los archivos que uno deja para que el otro lea:

- **`train.py`** — entrena una **versión** nueva del score. Ajusta
  (`fit_transform`) los escaladores de cada variable y los cortes
  Bajo/Medio/Alto de la versión de `config.yml → scoring.version`, en
  `models/score/{version}/*.pkl` y `configs/scoring/cortes_scoring_{version}.json`.
  Todo lo demás que produce (raw, analytic, scoring, priorización) queda en
  `data/train/{version}/`, sin tocar lo que ya entregó la inferencia. Se niega
  a correr si esa versión ya existe: para reentrenar se sube la versión.
- **`inference.py`** — puntúa sin reentrenar. Carga los `.pkl` y los cortes
  de la versión vigente (`.transform()`, nunca `.fit()`),
  separa qué cédulas de la corrida ya estaban etiquetadas en el histórico de
  qué cédulas son genuinamente nuevas, y solo esas últimas son las que de
  verdad necesitan gestión. Se corre periódicamente (ej. cada mes) usando
  siempre el mismo modelo congelado, para que el score sea comparable entre
  corridas.

Ambos comparten las etapas 1 (extracción) y 2 (transformación) — literalmente
importan las mismas funciones — y solo divergen en la etapa de scoring.

```
                 ┌─────────────────────┐
  BodegaCorp. →  │  1. EXTRACCIÓN       │  src/extraccion/extract_raw.py
  GCC (Oracle) → │  extract_raw()       │  → data/raw/{periodo}/*.parquet
  CLV.xlsx     → │                      │
                 └──────────┬──────────┘
                            │ dict[str, DataFrame]
                            ▼
                 ┌─────────────────────┐
                 │  2. TRANSFORMACIÓN   │  src/transformacion/build_analytic_score.py
                 │  build_analytic_score│  → data/analytic/{periodo}/analytic_score_base.parquet
                 └──────────┬──────────┘
                            │ 1 fila por Id, variables D1-D5 crudas
                            ▼
              ┌─────────────┴─────────────┐
              ▼                            ▼
  ┌──────────────────────┐    ┌──────────────────────────┐
  │ 3a. ENTRENAMIENTO     │    │ 3b. INFERENCIA            │
  │ src/Scoring/          │    │ src/Scoring/inference.py  │
  │ build_score.py        │    │ ejecutar_inferencia()     │
  │ (fit_transform)        │    │ (transform, sin reajustar)│
  │ → data/train/{v}/      │    │ → data/scoring/{periodo}/ │
  │   scoring_inactivos    │    │   scoring_inactivos +     │
  │   .parquet             │    │   scoring_nuevos_inactivos│
  │ → models/score/{v}/    │    │ → data/scoring/           │
  │ → configs/scoring/     │    │   scoring_inactivos.parquet│
  │   cortes_scoring_{v}   │    │   (maestro, actualizado)  │
  └──────────┬────────────┘    │ → paths.out_score/{periodo}│
             │                 │   (.xlsx, ruta compartida) │
             ▼                 └──────────┬────────────────┘
  ┌──────────────────────┐                ▼
  │ 4. CARACTERIZACIÓN    │     priorización Alto (zoom_alta.py,
  │ caracterizar_score.py │     reusado, ya era transform-only)
  │ → reports/scoring/    │     → data/scoring/{periodo}/ +
  │   (diagnóstico)       │       out_score/{periodo}/
  └──────────┬────────────┘       priorizacion_numerica_alta.xlsx
             ▼
  ┌──────────────────────┐
  │ 5. PRIORIZACIÓN ALTA  │
  │ (entrenamiento)        │
  │ zoom_alta.build_zoom() │
  │ → data/train/{v}/      │
  │   priorizacion_alta   │
  │   .xlsx                │
  └──────────────────────┘
```

### Estructura de carpetas

```
Inactivos/
├── train.py                    # entry point: entrenamiento (pasos 1,2,3a,4,5)
├── inference.py                 # entry point: inferencia (pasos 1,2,3b)
├── configs/
│   ├── config.yml               # pesos, versión vigente del score, rutas, orden CLV
│   └── scoring/cortes_scoring_{version}.json   # cortes Bajo/Medio/Alto por versión (los escribe train.py)
├── sql/                          # queries contra BodegaCorporativa/GCC (.sql, placeholders '?'/{ids})
├── src/
│   ├── extraccion/extract_raw.py
│   ├── transformacion/build_analytic_score.py
│   ├── Scoring/
│   │   ├── build_score.py        # entrenamiento: fit_transform + score_compromiso
│   │   ├── inference.py          # inferencia: transform (sin reajustar) + split nuevos + priorización
│   │   ├── caracterizar_score.py # diagnóstico post-hoc (solo entrenamiento)
│   │   └── zoom_alta.py          # mini-score de priorización para categoria_score == 'Alto'
│   └── utils/
│       ├── config.py             # load_config() -> lee configs/config.yml
│       └── helpers.py            # periodo_mas_cercano(), cortes y rutas por versión -- compartidos
├── data/
│   ├── Insumos/                 # único Excel fuente que queda: CLV.xlsx
│   ├── raw/{periodo}/            # snapshots parquet de cada corrida de inferencia, YYYYMM
│   ├── analytic/{periodo}/       # base analítica de inferencia (variables D1-D5 crudas), YYYYMM
│   ├── scoring/
│   │   ├── scoring_inactivos.parquet      # MAESTRO histórico acumulado (lo actualiza inference.py)
│   │   └── {periodo}/                     # salidas de inference.py para esa corrida
│   └── train/{version}/          # todo lo de un entrenamiento: raw/, analytic/,
│                                 #   scoring_inactivos.parquet, priorizacion_alta.xlsx
├── models/score/{version}/       # MinMaxScaler/OrdinalEncoder (.pkl) de esa versión, uno por variable
└── reports/scoring/{version}/    # histograma, resúmenes por categoría (diagnóstico de train.py)
```

---

## Las 5 dimensiones del score

Definidas en `calcular_scoring()` (`src/Scoring/build_score.py`) — y su
espejo transform-only `inferir_scoring()` en `src/Scoring/inference.py` —,
pesos en `configs/config.yml → scoring` (ya suman 1.0):

| Dim. | Nombre | Peso | Variables |
|------|--------|------|-----------|
| D1 | Severidad de inactividad | 0.22 | Tiempo inactivo, % mora GECC 6M, recaudo promedio 6M, intenciones de retiro último año |
| D2 | Enganche | 0.22 | N° productos, N° empresas del grupo con vínculo, valor perseverancia/ingresos, total eventos/usos, valor capitalizado |
| D3 | Recencia | 0.22 | Distancia a última reactivación, recencia de última mejora de producto, turnos en oficina 12M, PQR último año |
| D4 | Vínculo histórico | 0.22 | Cuotas pagadas vs antigüedad, reactivaciones previas, rango CLV, saldo de aportes, cercanía a perseverancia |
| D5 | Externo | 0.12 | 3 alertas binarias de centrales/hábito de pago externo (invertidas: sin alertas = mejor) |

Cada dimensión es el **promedio simple** (ignorando NaN) de sus variables ya
normalizadas a [0,1]; el score final es la combinación lineal ponderada,
escalada a [0,100]. En entrenamiento, `categorizar_score()` corta la
distribución en Bajo/Medio/Alto usando `media ± 0.5·std` del lote corrido y
escribe `cortes_scoring_{version}.json`; en inferencia, `etiquetar_categoria()`
usa los cortes de la versión vigente tal cual, sin recalcularlos.

> `Oferta_Reactivacion_Disponible` (antes parte de D1) se eliminó del score:
> ya no se consulta ni se pondera — D1 opera con 4 variables, no 5.

### Normalización

Cada variable pasa por una de estas familias de función, según su forma de
distribución — con dos implementaciones paralelas, una por escalador
(`normalizar_*` en `build_score.py`, ajusta y persiste) y otra por transform
(`inferir_*` en `inference.py`, carga y aplica sin reajustar):

- `normalizar_continua` / `inferir_continua` — MinMax con winsorización a
  percentiles [5,95].
- `normalizar_log` / `inferir_log` — igual, pero con `log1p` previo
  (variables sesgadas: valor capitalizado, saldo, distancia de reactivación,
  etc.).
- `normalizar_zero_inflated` / `inferir_zero_inflated` — para variables con
  masa de ceros estructural (ceros → 0.0 directo; positivos → `(0.1, 1.0]`
  tras log+MinMax).
- `normalizar_categoricas` / `inferir_categoricas` — `OrdinalEncoder` + MinMax,
  usada solo para `Clv`.

El recorte a percentiles [5,95] (`clampear_percentil`) solo se calcula al
entrenar. Al inferir (y en el zoom, que solo transforma) se recorta con los
límites que quedaron guardados en el escalador, `recortar_a_entrenamiento`,
no con los percentiles del lote: si no, el score de una cédula dependería de
cómo vino el resto del mes.

Solo las funciones `normalizar_*` (entrenamiento) escriben en
`models/score/{version}/` — las `inferir_*` solo leen de ahí. Si la versión de
`config.yml` no está entrenada, `inference.py` falla con `FileNotFoundError`
explícito (no hay artefacto que cargar).

---

## Decisiones clave (no obvias desde el código)

- **Población base ya no es un Excel**: `extract_inactivos()` consulta
  BodegaCorporativa directo (tabla `factasociatividad`, estado = Inactivo) para
  el mes en curso — reemplaza al viejo `Poblacion_Inactivos.xlsx`. Cada
  corrida trae un único periodo (el mes actual), no varios a la vez.
- **Versiones del score**: escaladores + cortes se entrenan y se usan
  juntos, como una versión (`config.yml → scoring.version`). Cortes de una
  versión con escaladores de otra darían categorías inconsistentes. Cada
  fila que puntúa `inference.py` lleva la columna `version_score`.
  - **v1**: entrenada con 202606. `Perseverancia_Cerca` venía del Excel
    `features_inactivos` con la lógica invertida (`>60` meses → 1) y las
    alertas de D5 de `Enrequecimiento_360.xlsx`.
  - **v2**: entrenada con 202609 re-extraído con las fuentes corregidas
    (360 histórica, particiones completas). `Perseverancia_Cerca = 1` si
    faltan entre 0 y 90 meses (negativo = ya perseveró → 0; nulo = sin SIPAS).
- **Rezago de 1 mes en el periodo de consulta, con retroceso**:
  `Numcantidadproductos`, tenencia, `v_360`, `demografica` y factura/recaudo
  GECC se consultan en el mes *anterior* al periodo objetivo (ej. población
  en 202609 → se consulta 202608). BI no siempre tiene ese mes cargado el día
  que se corre (el 1 de septiembre factura de agosto venía vacía y tenencia a
  medias), así que `resolver_particion()` verifica antes que la partición
  tenga al menos el 90% de las cédulas del mes previo y, si no, retrocede un
  mes (máx. 2; si ninguna sirve, se detiene). Factura y recaudo se resuelven
  juntas. La transformación ancla sus ventanas (promedio 6M, recencia) al
  último mes que de verdad se extrajo (`_periodo_referencia`).
  `Distancia_ultima_reactivacion` y `Cantidad_Reactivaciones_Previas` sí usan
  el periodo objetivo tal cual, porque comparan contra el Excel maestro de
  reactivaciones, no contra BodegaCorporativa.
- **`v_360.sql` lee de `ConsultaIntegral360` (histórica), no de
  `ConsultaIntegral360_Diaria`**: la diaria solo guarda la última carga (el
  mes de referencia desaparece apenas entra la siguiente) y trae
  `Tipo_Cliente_Bancoomeva` vacío.
- **Solo queda un insumo Excel**: `CLV.xlsx` (`config.yml → paths_bases`) y el
  Excel maestro de reactivaciones históricas (`config.yml → paths.path_reac`,
  ruta de red), usado solo para `Distancia_ultima_reactivacion` y
  `Cantidad_Reactivaciones_Previas` — cédulas sin historial previo no son un
  error, caen en `Cantidad_Reactivaciones_Previas=0` y
  `Distancia_ultima_reactivacion` usa el fallback `Antiguedad*12`.
  `features_inactivos.xlsx`, `enriquecimiento_360.xlsx` y
  `consolidado_reactivaciones.xlsx` se **eliminaron**: sus variables ya están
  cubiertas por consultas SQL (`meses_inactivo.sql`, `sipas.sql`, `v_360.sql`,
  `pqrs.sql`, y una CTE de intenciones de retiro dentro de `demo.sql`).
- **`intenciones_retiro_1y` sale de una CTE en `demo.sql`**, no de un Excel:
  cuenta registros de `intencionesDeRetiro` en la misma ventana retrospectiva
  de 12 meses que usa el resto de `demografica`.
- **Vencido vs Cuota en `calcular_promedio_fac_rec`**: `Vencido_Mes` es un
  *stock* acumulado y `Cuota_Mes` un *flujo* mensual — no se suman
  directamente. `Pct_Mora` se calcula como
  `Vencido_Promedio / (Vencido_Promedio + Factura_Total_Promedio)` para evitar
  doble conteo y para que la severidad sea proporcional, no en pesos absolutos.
- **CLV.xlsx** es una copia del mismo archivo usado en `MCD-PDG1`, no
  específico de esta población — cédulas sin cruce quedan con `Clv=NaN`, que
  el encoder trata como categoría desconocida (score 0 en esa variable).
- **Carpeta por periodo (YYYYMM) en cada etapa**: `data/raw/`,
  `data/analytic/` y las salidas de `inference.py` en `data/scoring/` se
  organizan como `{etapa}/{periodo}/`, no como un archivo plano con fecha de
  extracción en el nombre. Cada corrida sobrescribe la carpeta de su propio
  periodo (correr dos veces en el mismo mes pisa la anterior a propósito).
- **El maestro (`data/scoring/scoring_inactivos.parquet`) crece por
  apéndice, no se recrea**: `inference.py` decide "cédula nueva" comparando
  solo por `Id` contra este archivo (sin importar el `Periodo`), y al
  terminar hace `pd.concat(maestro, nuevos)` y lo sobrescribe. Así una
  cédula que sigue inactiva mes a mes y ya se etiquetó una vez no vuelve a
  contar como "nueva" en la corrida siguiente. Es idempotente por periodo:
  correr el mismo mes dos veces reemplaza las filas de ese periodo en vez de
  duplicarlas o dejar "nuevos" vacío.
- **`fit_transform` (entrenar) vs `transform` (inferir), ya separados a nivel
  de pipeline completo**: `build_score.py`/`train.py` reajustan los
  escaladores y los cortes en cada corrida (a propósito: es la etapa de
  entrenamiento). `inference.py` es el módulo que carga esos mismos
  artefactos y solo transforma — nunca reajusta, nunca reescribe
  `models/score/` ni los cortes. Antes de que existiera
  `inference.py`, esta separación solo existía en `zoom_alta.py`; ahora
  aplica a todo el pipeline de scoring.
- **La priorización numérica de `zoom_alta.py` sigue la misma metodología
  que el score**: `zoom()` usa los escaladores de la versión vigente, y los
  terciles de `ZoomAlta` que separan prioridad 1/2/3 se calculan una sola vez
  en `train.py` (`entrenar_cortes_zoom`, sobre las Alto del lote de
  entrenamiento) y se guardan en `configs/scoring/cortes_zoom_{version}.json`.
  En inferencia `agruparzoom()` los aplica congelados, así que la prioridad es
  comparable entre corridas (ya no hay terciles exactos de 1/3 en cada lote).
  Los de v1 se reconstruyeron de la priorización de junio (reproducen el
  100% de sus asignaciones).
- **`MinMax_oferta_reactivacion.pkl`** (en `models/score/v1/`) es un artefacto
  huérfano de una versión anterior del score — ninguna función actual lo
  carga ni lo usa. No se borró para no perder historial, pero no hace nada.

---

## Requisitos y setup

- `.env` en la raíz con:
  - `CON_BI` — cadena/DSN `pyodbc` a BodegaCorporativa (SQL Server).
  - `GCC_CON` — cadena/DSN `pyodbc` a GCC (Oracle, usada solo por
    `meses_inactivo.sql`).
- Entorno virtual `.venv313` ya creado en el repo
  (`.venv313\Scripts\python.exe`). Dependencias en `requeriments.txt`
  (pandas, scikit-learn, pyodbc, joblib, pyyaml, python-dotenv, matplotlib,
  pyarrow, openpyxl, entre otras).
- `configs/config.yml → paths` con dos rutas UNC llenas:
  - `path_reac` — Excel maestro de reactivaciones históricas.
  - `out_score` — carpeta de red compartida donde `inference.py` deja las
    copias en `.xlsx` para el negocio (`{out_score}/{periodo}/`).
- Para `inference.py`: que la versión de `config.yml → scoring.version` ya
  esté entrenada (que existan `models/score/{version}/*.pkl` y
  `configs/scoring/cortes_scoring_{version}.json`).

## Cómo correr

Entrenar una versión nueva (subir antes `scoring.version` en `config.yml`,
ej. `v2` → `v3`; `train.py` no pisa una versión que ya existe):

```powershell
.venv313\Scripts\python.exe train.py
```

Corre extracción → transformación → scoring (fit_transform) →
caracterización → priorización de altos. Genera `data/train/{version}/`
(`raw/{periodo}/`, `analytic/{periodo}/`, `scoring_inactivos.parquet`,
`priorizacion_alta.xlsx`), `models/score/{version}/*.pkl`,
`configs/scoring/cortes_scoring_{version}.json` y los reportes de diagnóstico
en `reports/scoring/{version}/`. Desde ese momento `inference.py` usa esa
versión. Para volver a una anterior basta con cambiar `scoring.version`.

Inferir (usa el modelo ya entrenado, sin reajustar; correr periódicamente,
ej. cada mes):

```powershell
.venv313\Scripts\python.exe inference.py
```

Corre extracción → transformación → inferencia (transform, sin reentrenar).
Genera `data/raw/{periodo}/`, `data/analytic/{periodo}/`,
`data/scoring/{periodo}/scoring_inactivos.parquet` (población completa de
esta corrida), `data/scoring/{periodo}/scoring_nuevos_inactivos.parquet`
(solo cédulas nuevas), actualiza el maestro
`data/scoring/scoring_inactivos.parquet`, y deja
`priorizacion_numerica_alta.xlsx` (cédulas Alto de esta corrida) en
`data/scoring/{periodo}/` y en `{out_score}/{periodo}/`, junto con
`score_nuevos_inactivos.xlsx` en esta última.

---

## Estado actual y limitaciones conocidas

- `data/scoring/scoring_inactivos.parquet` (el maestro que usa
  `inference.py`) hoy solo tiene el periodo con el que se entrenó
  originalmente el modelo — la primera corrida real de `inference.py` va a
  marcar como "nueva" prácticamente toda la población inactiva del mes en
  curso. Es esperado, no un bug: de ahí en adelante el maestro ya empieza a
  filtrar de verdad.
- No hay todavía un mecanismo de reentrenamiento automático/periódico: correr
  `train.py` es una acción manual y deliberada, no algo que dispare
  `inference.py` por su cuenta.
