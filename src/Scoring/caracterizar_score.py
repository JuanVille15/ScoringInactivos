"""Caracterización post-hoc del scoring: valida la distribución de
score_compromiso y resume las variables crudas (mediana, moda) por categoría
Bajo/Medio/Alto.

No es parte del pipeline de cómputo (extracción -> transformación -> scoring);
se corre después de `build_score()` como diagnóstico, para revisar que la
primera corrida con datos reales tenga sentido antes de confiar en el score.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.helpers import periodo_mas_cercano

# Variables donde se fuerza mediana o media en 'valor_horizontal' en vez de la
# regla automática moda/media. La regla automática (moda si varía entre grupos,
# si no media) falla en variables continuas de alta cardinalidad: ahí la moda es
# el valor exacto que por azar se repitió una vez más que los demás, no un
# "valor típico" — puede salir 0 en dos de los tres grupos aunque casi nadie
# tenga 0 (ej. Vencido_Promedio_GECC_6M: pct_no_cero > 93% en los tres grupos,
# pero la moda automática daba 48.460 / 0 / 0). Esas variables se fuerzan aquí
# a mano en vez de confiar en la heurística.
FORZAR_MEDIANA = {"Tiempo_Inactividad_Meses"}
FORZAR_MEDIA = {"Vencido_Promedio_GECC_6M", "Pct_Mora_GECC_6M"}

# ─── Lectura de artefactos ────────────────────────────────────────────────────

def _leer_scoring(scoring_path: str | None = None) -> pd.DataFrame:
    """Lee el parquet de scoring final.

    Args:
        scoring_path: Ruta personalizada. Si es None, usa
            ``data/scoring/scoring_inactivos.parquet``.

    Returns:
        pd.DataFrame con Periodo, Id, dimensiones D1-D5, score_compromiso y
        categoria_score.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(scoring_path) if scoring_path else Path.cwd() / "data" / "scoring" / "scoring_inactivosv2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero build_score().")
    return pd.read_parquet(path, engine="pyarrow")


def _leer_cortes(cortes_path: str | None = None) -> dict:
    """Lee los cortes Bajo/Medio/Alto persistidos por `categorizar_score`.

    Args:
        cortes_path: Ruta personalizada. Si es None, usa
            ``configs/scoring/cortes_scoring.json``.

    Returns:
        dict con claves 'bajo', 'medio', 'alto', cada una [limite_inf, limite_sup].

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(cortes_path) if cortes_path else Path.cwd() / "configs" / "scoring" / "cortes_scoring.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero build_score().")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Distribución ──────────────────────────────────────────────────────────────

def graficar_distribucion_score(
    scoring_path: str | None = None,
    cortes_path: str | None = None,
    out_path: str | None = None,
) -> Path:
    """Grafica el histograma de score_compromiso con los cortes Bajo/Medio/Alto.

    Args:
        scoring_path: Ruta personalizada al parquet de scoring. Ver `_leer_scoring`.
        cortes_path: Ruta personalizada al json de cortes. Ver `_leer_cortes`.
        out_path: Ruta de salida del .jpg. Si es None, usa
            ``reports/scoring/distribucion_score.jpg``.

    Returns:
        Path del archivo .jpg exportado.
    """
    score = _leer_scoring(scoring_path)
    cortes = _leer_cortes(cortes_path)

    corte_bajo = cortes["bajo"][1]
    corte_alto = cortes["alto"][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(score["score_compromiso"], bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(corte_bajo, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Corte Bajo/Medio ({corte_bajo:.0f})")
    ax.axvline(corte_alto, color="#55A868", linestyle="--", linewidth=1.5,
               label=f"Corte Medio/Alto ({corte_alto:.0f})")
    ax.set_title(f"Distribución de score_compromiso (n={len(score):,})")
    ax.set_xlabel("score_compromiso [0-100]")
    ax.set_ylabel("Cantidad de cédulas")
    ax.legend()
    fig.tight_layout()

    path_salida = Path(out_path) if out_path else Path.cwd() / "reports" / "scoring" / "distribucion_scoreV2.jpg"
    path_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)

    print(f"Exportado: {path_salida}")
    return path_salida

# ─── Resumen por grupo ─────────────────────────────────────────────────────────

def resumen_variables_por_grupo(
    scoring_path: str | None = None,
    periodo: str | None = None,
    analytic_path: str | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Calcula mediana y moda de cada variable cruda, agrupadas por categoria_score.

    Cruza el scoring final (categoria_score) con la base analítica (variables
    crudas D1-D5 antes de normalizar) para ver qué caracteriza a cada grupo
    Bajo/Medio/Alto en sus propias unidades (meses, pesos, conteos), no en la
    escala [0,1] normalizada.

    Args:
        scoring_path: Ruta personalizada al parquet de scoring. Ver `_leer_scoring`.
        periodo: Periodo (YYYYMM) cuya base analítica leer, ej. '202608'. Si es
            None, se toma la carpeta de periodo más reciente bajo
            ``data/analytic/``. Se ignora si se pasa `analytic_path`.
        analytic_path: Ruta personalizada y completa al parquet de la base
            analítica (se usa tal cual, sin resolver periodo). Si es None, se
            resuelve como ``data/analytic/{periodo}/analytic_score_base.parquet``.
        out_path: Ruta de salida del .csv. Si es None, usa
            ``reports/scoring/resumen_variables_por_grupo.csv``.

    Returns:
        pd.DataFrame con columnas ['variable', 'categoria_score', 'mediana',
        'media', 'pct_no_cero', 'moda', 'valor_horizontal', 'conteo']. Una
        fila por combinación variable x categoría. 'mediana', 'media' y
        'pct_no_cero' quedan en None para variables no numéricas (ej. Clv).
        'media' y 'pct_no_cero' se agregan porque varias variables (ej.
        recaudo/factura promedio, PQR, intenciones de retiro) están
        fuertemente zero-inflated: la mediana da 0 en los tres grupos aunque
        la variable sí discrimine — la media y el % de valores distintos de
        cero sí lo muestran.

        'valor_horizontal' es el valor que usa `resumen_variables_por_grupo_horizontal`
        para cada celda: moda para variables categóricas/booleanas (Clv, las
        3 alertas D5, Perseverancia_Cerca — con ≤2 valores posibles la moda
        es la lectura natural), y para el resto (numéricas con más de 2
        valores) moda si esta varía entre Bajo/Medio/Alto (discrimina), o
        media si la moda es igual en los tres grupos (variable zero-inflated
        o con un valor dominante — ahí la moda no aporta nada y la media sí).

    Raises:
        FileNotFoundError: Si el parquet de scoring o el de la base analítica
            no existen.
    """
    score = _leer_scoring(scoring_path)[[
        "Periodo", "Id", "d1_severidad", "d2_enganche", "d3_recencia",
        "d4_vinculo", "d5_externo", "score_compromiso", "categoria_score",
    ]]

    if analytic_path is not None:
        path_analytic = Path(analytic_path)
    else:
        raiz_analytic = Path.cwd() / "data" / "analytic"
        periodo_resuelto = periodo or periodo_mas_cercano(raiz_analytic, "analytic_score_base.parquet")
        path_analytic = raiz_analytic / periodo_resuelto / "analytic_score_base.parquet"

    if not path_analytic.exists():
        raise FileNotFoundError(f"No existe {path_analytic}. Corre primero build_analytic_score().")
    analytic = pd.read_parquet(path_analytic, engine="pyarrow")

    df = score.merge(right=analytic, how="left", on=["Periodo", "Id"])

    columnas_variables = [c for c in analytic.columns if c not in ("Periodo", "Id")] + [
        "d1_severidad", "d2_enganche", "d3_recencia", "d4_vinculo", "d5_externo",
        "score_compromiso",
    ]

    filas = []
    for variable in columnas_variables:
        es_numerica = pd.api.types.is_numeric_dtype(df[variable])
        es_booleana_o_categorica = (not es_numerica) or (df[variable].nunique(dropna=True) <= 2)

        agg_kwargs = {
            "conteo": (variable, "count"),
            "moda": (variable, lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None),
        }
        if es_numerica:
            agg_kwargs["mediana"] = (variable, "median")
            agg_kwargs["media"] = (variable, "mean")
            agg_kwargs["pct_no_cero"] = (variable, lambda s: (s.fillna(0) != 0).mean())

        resumen_var = df.groupby("categoria_score", observed=True).agg(**agg_kwargs).reset_index()
        if not es_numerica:
            resumen_var["mediana"] = None
            resumen_var["media"] = None
            resumen_var["pct_no_cero"] = None

        if variable in FORZAR_MEDIANA:
            resumen_var["valor_horizontal"] = resumen_var["mediana"]
        elif variable in FORZAR_MEDIA:
            resumen_var["valor_horizontal"] = resumen_var["media"]
        elif es_booleana_o_categorica:
            resumen_var["valor_horizontal"] = resumen_var["moda"]
        else:
            moda_discrimina = resumen_var["moda"].nunique(dropna=True) > 1
            resumen_var["valor_horizontal"] = resumen_var["moda"] if moda_discrimina else resumen_var["media"]

        resumen_var["variable"] = variable
        filas.append(resumen_var[[
            "variable", "categoria_score", "mediana", "media", "pct_no_cero",
            "moda", "valor_horizontal", "conteo",
        ]])

    resumen = pd.concat(filas, ignore_index=True)

    path_salida = Path(out_path) if out_path else Path.cwd() / "reports" / "scoring" / "resumen_variables_por_grupoV2.csv"
    path_salida.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(path_salida, index=False, encoding="utf-8")

    print(f"Exportado: {path_salida}")
    return resumen

# ─── Resumen horizontal ────────────────────────────────────────────────────────

def resumen_variables_por_grupo_horizontal(
    resumen: pd.DataFrame | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Pivota `resumen_variables_por_grupo` a formato ancho: filas=categoria_score,
    columnas=variable, valor='valor_horizontal'.

    'valor_horizontal' (calculado en `resumen_variables_por_grupo`) usa moda
    para variables categóricas/booleanas (Clv, alertas D5, Perseverancia_Cerca)
    y, para el resto, moda si esta varía entre grupos (discrimina) o media si
    la moda es igual en Bajo/Medio/Alto (variable zero-inflada o con un valor
    dominante, donde la moda no aportaría nada).

    Args:
        resumen: Salida de `resumen_variables_por_grupo` (formato largo). Si es
            None, la calcula llamando a `resumen_variables_por_grupo()`.
        out_path: Ruta de salida del .csv. Si es None, usa
            ``reports/scoring/resumen_variables_por_grupo_horizontal.csv``.

    Returns:
        pd.DataFrame con índice categoria_score (Bajo/Medio/Alto), columna
        'n_cedulas' (cantidad de cédulas del grupo) seguida de una columna
        por variable, valores en 'valor_horizontal'.
    """
    if resumen is None:
        resumen = resumen_variables_por_grupo()

    horizontal = resumen.pivot(index="categoria_score", columns="variable", values="valor_horizontal")
    horizontal = horizontal.reindex(["Bajo", "Medio", "Alto"])

    n_cedulas = (
        resumen[resumen["variable"] == "score_compromiso"]
        .set_index("categoria_score")["conteo"]
        .reindex(["Bajo", "Medio", "Alto"])
    )
    horizontal.insert(0, "n_cedulas", n_cedulas)

    path_salida = (
        Path(out_path) if out_path
        else Path.cwd() / "reports" / "scoring" / "resumen_variables_por_grupo_horizontalV2.csv"
    )
    path_salida.parent.mkdir(parents=True, exist_ok=True)
    horizontal.to_csv(path_salida, encoding="utf-8")

    print(f"Exportado: {path_salida}")
    return horizontal

# ─── Orquestadora ─────────────────────────────────────────────────────────────

def caracterizar_score(periodo: str | None = None) -> None:
    """Orquesta el diagnóstico post-hoc del scoring: distribución + resumen por grupo.

    Función de entrada principal del módulo. Corre después de `build_score()`.

    Flujo de ejecución:
        1. `graficar_distribucion_score` -> histograma de score_compromiso con
           los cortes Bajo/Medio/Alto, exportado a
           ``reports/scoring/distribucion_score.jpg``.
        2. `resumen_variables_por_grupo` -> mediana/media/moda de cada variable
           cruda por categoria_score (formato largo), exportado a
           ``reports/scoring/resumen_variables_por_grupo.csv``.
        3. `resumen_variables_por_grupo_horizontal` -> el mismo resumen pivotado
           a formato ancho (filas=categoria_score, columnas=variable, valor=
           media), exportado a
           ``reports/scoring/resumen_variables_por_grupo_horizontal.csv``.

    Args:
        periodo: Periodo (YYYYMM) cuya base analítica leer, ej. '202608'. Si es
            None, se toma la carpeta de periodo más reciente en ``data/analytic/``.

    Raises:
        FileNotFoundError: Si `data/scoring/scoring_inactivos.parquet`,
            `configs/scoring/cortes_scoring.json` o la corrida de
            `data/analytic/{periodo}/analytic_score_base.parquet` no existen
            (correr primero build_analytic_score() y build_score()).
    """
    print("Caracterizando scoring...")
    graficar_distribucion_score()
    resumen = resumen_variables_por_grupo(periodo=periodo)
    resumen_variables_por_grupo_horizontal(resumen=resumen)
    print("Caracterización -- Finalizada")
