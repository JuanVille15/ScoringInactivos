"""Scoring D1-D5 (severidad, enganche, recencia, vínculo, externo) para la
población de inactivos. Los pesos (severidad/enganche/recencia/vinculo/externo)
ya suman 1.0 en config.yml, así que el score final queda directamente en
escala completa [0, 100].
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from src.utils.config import load_config
from src.utils.helpers import periodo_mas_cercano
import joblib
import json

# ─── Extracción ───────────────────────────────────────────────────────────────

def xtr_columnas_necesarias(
    periodo: str | None = None,
    analytic_path: str | None = None,
) -> pd.DataFrame:
    """Lee desde la base analítica solo las columnas requeridas por el scoring D1-D5.

    Args:
        periodo: Periodo (YYYYMM) a leer, ej. '202608'. Si es None, se toma
            la carpeta de periodo más reciente bajo ``data/analytic/``. Se
            ignora si se pasa `analytic_path`.
        analytic_path: Ruta personalizada y completa al parquet de la base
            analítica (se usa tal cual, sin resolver periodo). Si es None, se
            resuelve como ``data/analytic/{periodo}/analytic_score_base.parquet``.

    Returns:
        pd.DataFrame: Subconjunto con Periodo, Id y las variables de D1-D5.

    Raises:
        FileNotFoundError: Si el archivo resuelto no existe (o, sin `periodo`
            ni `analytic_path`, si no hay ninguna corrida de transformación).
    """
    if analytic_path is not None:
        path_base = Path(analytic_path)
    else:
        raiz_analytic = Path.cwd() / "data" / "analytic"
        periodo_resuelto = periodo or periodo_mas_cercano(raiz_analytic, "analytic_score_base.parquet")
        path_base = raiz_analytic / periodo_resuelto / "analytic_score_base.parquet"

    if not path_base.exists():
        raise FileNotFoundError(f"No existe {path_base}. Corre primero el módulo de transformación.")

    return pd.read_parquet(
        path=path_base,
        engine="pyarrow",
        columns=[
            "Periodo", "Id",
            # D1 · Severidad inactividad
            "Tiempo_Inactividad_Meses", "Pct_Mora_GECC_6M",
            "Recaudo_Total_Promedio_GECC_6M",
            "intenciones_retiro_1y",
            # D2 · Enganche
            "Numcantidadproductos", "Cantidad_empresas", "Valor_perseverancia_vs_ingresos",
            "Total_Eventos", "Valor_Capitalizado",
            # D3 · Recencia
            "Distancia_ultima_reactivacion", "Recencia_ultimo_producto",
            "Turnos_En_Oficinas_Total_Ult12Meses", "Cantidad_pqr_ultimo_anno",
            # D4 · Vínculo histórico
            "Cuotas_pagadas_vs_antiguedad", "Cantidad_Reactivaciones_Previas",
            "Clv", "Saldoaportes", "Perseverancia_Cerca",
            # D5 · Externo
            "Alerta_Habito_Pago_Externo", "Alerta_Estado_Creditos_Externos",
            "Alerta_Capacidad_Pago_Externo",
        ],
    )

# ─── Helpers de normalización ─────────────────────────────────────────────────

def clampear_percentil(series: pd.Series, p_low: int = 5, p_high: int = 95) -> pd.Series:
    """Recorta una serie numérica al rango [p_low, p_high] para atenuar outliers.

    Args:
        series (pd.Series): Serie numérica a recortar. Puede contener NaN.
        p_low (int): Percentil inferior de corte. Por defecto 5.
        p_high (int): Percentil superior de corte. Por defecto 95.

    Returns:
        pd.Series: Serie recortada al intervalo [quantile(p_low), quantile(p_high)].
    """
    low  = series.quantile(p_low  / 100)
    high = series.quantile(p_high / 100)
    return series.clip(lower=low, upper=high)


def normalizar_categoricas(series: pd.Series, orden: list, name: str) -> pd.Series:
    """Codifica una variable ordinal categórica al rango continuo [0, 1].

    Categorías no vistas reciben el centinela -1, que MinMaxScaler(clip=True)
    lleva a 0.0 (score mínimo, decisión conservadora).

    Args:
        series (pd.Series): Serie con la variable categórica ordinal a normalizar.
        orden (list): Categorías ordenadas de menor a mayor valor semántico.
        name (str): Nombre del artefacto (ej. 'clv' -> OrdinalEncoder_clv.pkl).

    Returns:
        pd.Series: Serie normalizada en [0, 1].

    Note:
        Persiste ``OrdinalEncoder_{name}.pkl`` y ``MinMax_{name}.pkl`` en models/score/.
    """
    s = series.copy()

    enc = OrdinalEncoder(
        categories=[orden],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    serie_codificada = enc.fit_transform(s.to_numpy().reshape(-1, 1)).flatten()

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    serie_normalizada = scaler.fit_transform(serie_codificada.reshape(-1, 1)).flatten()

    joblib.dump(enc,    Path(f"models/score/OrdinalEncoder_{name}.pkl"))
    joblib.dump(scaler, Path(f"models/score/MinMax_{name}.pkl"))

    return pd.Series(serie_normalizada, index=s.index)


def normalizar_continua(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Normaliza una variable continua al rango [0, 1] con winsorización previa.

    Args:
        series (pd.Series): Serie numérica continua a normalizar.
        name (str): Nombre del artefacto (ej. 'saldo' -> MinMax_saldo.pkl).
        invertir (bool): Si True, aplica 1 - score. Usar cuando mayor valor
            original representa peor comportamiento. Por defecto False.

    Returns:
        pd.Series: Serie normalizada en [0, 1].

    Note:
        Persiste ``MinMax_{name}.pkl`` en models/score/.
    """
    s = clampear_percentil(series)

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    s_scaled = scaler.fit_transform(s.to_numpy().reshape(-1, 1)).flatten()

    joblib.dump(scaler, Path(f"models/score/MinMax_{name}.pkl"))

    serie_salida = pd.Series(data=s_scaled, index=s.index)
    return 1 - serie_salida if invertir else serie_salida


def normalizar_log(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Normaliza una variable de distribución sesgada mediante log1p + MinMax.

    Args:
        series (pd.Series): Serie numérica no negativa con distribución sesgada.
        name (str): Nombre del artefacto (ej. 'dist_reac' -> MinMax_dist_reac.pkl).
        invertir (bool): Si True, aplica 1 - score. Por defecto False.

    Returns:
        pd.Series: Serie normalizada en [0, 1].

    Note:
        Persiste ``MinMax_{name}.pkl`` en models/score/, ajustado en escala logarítmica.
    """
    s = np.log1p(clampear_percentil(series))

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    s_scaled = scaler.fit_transform(s.to_numpy().reshape(-1, 1)).flatten()  # type: ignore

    joblib.dump(scaler, Path(f"models/score/MinMax_{name}.pkl"))

    serie_salida = pd.Series(data=s_scaled, index=series.index)
    return 1 - serie_salida if invertir else serie_salida


def normalizar_zero_inflated(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Normaliza una variable con exceso estructural de ceros al rango [0, 1].

    Ceros/NaN -> 0.0 (sin actividad). Positivos -> (0.1, 1.0] tras winsorización,
    log1p y MinMax, ajustado solo sobre los valores positivos.

    Args:
        series (pd.Series): Serie numérica con masa de ceros.
        name (str): Nombre del artefacto (ej. 'saldo' -> MinMax_saldo.pkl).
        invertir (bool): Si True, aplica 1 - score. Por defecto False.

    Returns:
        pd.Series: Serie normalizada en [0, 1].

    Note:
        Persiste ``MinMax_{name}.pkl`` en models/score/, ajustado solo sobre positivos.
    """
    s = series.fillna(0).copy()

    mask_pos = s > 0
    s_norm = pd.Series(0.0, index=s.index)

    if mask_pos.sum() > 0:
        s_pos = np.log1p(clampear_percentil(s[mask_pos]))

        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        s_pos_scaled = scaler.fit_transform(s_pos.to_numpy().reshape(-1, 1)).flatten()  # type: ignore

        joblib.dump(scaler, Path(f"models/score/MinMax_{name}.pkl"))

        s_norm[mask_pos] = 0.1 + s_pos_scaled * 0.9  # type: ignore[operator]

    return 1 - s_norm if invertir else s_norm


def score_dimension(vals_dimension: list[pd.Series]) -> pd.Series:
    """Agrega las variables normalizadas de una dimensión como promedio simple.

    Args:
        vals_dimension (list[pd.Series]): Series ya normalizadas en [0, 1],
            todas con el mismo índice.

    Returns:
        pd.Series: Score de la dimensión en [0, 1], promedio ignorando NaN.
    """
    return pd.concat(vals_dimension, axis=1).mean(axis=1, skipna=True)


# ─── Pipeline de scoring ──────────────────────────────────────────────────────

def calcular_scoring(
    df: pd.DataFrame,
    orden_clv: list,
    pesos: dict,
) -> pd.DataFrame:
    """Construye las dimensiones D1-D5 y el score final ponderado [0, 100].

    Cada dimensión se calcula como el promedio simple de sus variables
    normalizadas (todas en [0, 1], ignorando NaN). El score final es una
    combinación lineal ponderada de las cinco dimensiones, escalada a [0, 100].

    Dimensiones y variables:
        - D1 Severidad (peso `pesos['severidad']`): Tiempo_Inactividad_Meses,
          Pct_Mora_GECC_6M, Recaudo_Total_Promedio_GECC_6M,
          intenciones_retiro_1y.
        - D2 Enganche  (peso `pesos['enganche']`): Numcantidadproductos,
          Cantidad_empresas, Valor_perseverancia_vs_ingresos, Total_Eventos,
          Valor_Capitalizado.
        - D3 Recencia  (peso `pesos['recencia']`): Distancia_ultima_reactivacion,
          Recencia_ultimo_producto, Turnos_En_Oficinas_Total_Ult12Meses,
          Cantidad_pqr_ultimo_anno.
        - D4 Vínculo   (peso `pesos['vinculo']`): Cuotas_pagadas_vs_antiguedad,
          Cantidad_Reactivaciones_Previas, Clv, Saldoaportes, Perseverancia_Cerca.
        - D5 Externo   (peso `pesos['externo']`): Alerta_Habito_Pago_Externo,
          Alerta_Estado_Creditos_Externos, Alerta_Capacidad_Pago_Externo.
          Las 3 son binarias (1 = tiene alerta, mal indicio) y ya viven en
          {0, 1}: no se escalan, se invierten una a una (1 - alerta) antes de
          promediar, para que "sin alertas" quede en 1.0 (mejor) y "con las
          3 alertas" en 0.0 (peor).

    Args:
        df (pd.DataFrame): DataFrame con Periodo, Id y las variables de D1-D5.
        orden_clv (list): Categorías de Clv ordenadas de menor a mayor valor.
            Leído desde config.yml -> scoring.orden_clv.
        pesos (dict): Pesos con claves 'severidad', 'enganche', 'recencia',
            'vinculo', 'externo' sumando 1.0 (config.yml -> scoring).

    Returns:
        pd.DataFrame: Tabla con columnas ['Periodo', 'Id', 'd1_severidad',
            'd2_enganche', 'd3_recencia', 'd4_vinculo', 'd5_externo',
            'score_compromiso']. Las dimensiones están en [0, 1];
            score_compromiso en [0, 100].

    Note:
        Persiste artefactos en models/score/ como efecto secundario: un
        MinMaxScaler por variable continua/log/zero_inflated, más OrdinalEncoder
        y MinMaxScaler para Clv. `Perseverancia_Cerca` y las 3 alertas de D5
        ya viven en {0, 1} y no generan artefacto (no se escalan).
    """
    df = df.copy()

    df["Numcantidadproductos"] = df["Numcantidadproductos"].fillna(0)
    df["Cantidad_empresas"]    = df["Cantidad_empresas"].fillna(0)

    # D1 · SEVERIDAD INACTIVIDAD
    v_tiempo_inact = normalizar_continua(df["Tiempo_Inactividad_Meses"],              name="tiempo_inactividad", invertir=True)
    v_saldo_deuda  = normalizar_continua(df["Pct_Mora_GECC_6M"],                      name="saldo_deuda",       invertir=True)
    v_pagos_parc   = normalizar_zero_inflated(df["Recaudo_Total_Promedio_GECC_6M"],    name="pagos_parciales")
    v_intenc_retiro = normalizar_zero_inflated(df["intenciones_retiro_1y"],            name="intencion_retiro",  invertir=True)
    d1 = score_dimension([v_tiempo_inact, v_saldo_deuda, v_pagos_parc, v_intenc_retiro])

    # D2 · ENGANCHE
    v_numprod    = normalizar_continua(df["Numcantidadproductos"],                    name="num_productos")
    v_empresas   = normalizar_continua(df["Cantidad_empresas"],                       name="num_empresas")
    v_proteccion = normalizar_log(df["Valor_perseverancia_vs_ingresos"],              name="proteccion")
    v_n_usos     = normalizar_log(df["Total_Eventos"],                                name="n_usos")
    v_valor_cap  = normalizar_log(df["Valor_Capitalizado"],                           name="valor_capitalizado")
    d2 = score_dimension([v_numprod, v_empresas, v_proteccion, v_n_usos, v_valor_cap])

    # D3 · RECENCIA
    v_dist_reac    = normalizar_log(df["Distancia_ultima_reactivacion"],              name="dist_reac")
    v_rec_prod     = normalizar_continua(df["Recencia_ultimo_producto"],              name="rec_prod",      invertir=True)
    v_turnos_ofic  = normalizar_zero_inflated(df["Turnos_En_Oficinas_Total_Ult12Meses"], name="turnos_oficina")
    v_pqr          = normalizar_zero_inflated(df["Cantidad_pqr_ultimo_anno"],          name="pqr",            invertir=True)
    d3 = score_dimension([v_dist_reac, v_rec_prod, v_turnos_ofic, v_pqr])

    # D4 · VÍNCULO HISTÓRICO
    v_cuotas_ant = normalizar_continua(df["Cuotas_pagadas_vs_antiguedad"],            name="cuotas_ant")
    v_reac_prev  = normalizar_continua(df["Cantidad_Reactivaciones_Previas"],         name="reac_prev",     invertir=True)
    v_clv        = normalizar_categoricas(df["Clv"], orden_clv,                       name="clv")
    v_saldo      = normalizar_log(df["Saldoaportes"],                                 name="saldo")
    v_persev     = df["Perseverancia_Cerca"]
    d4 = score_dimension([v_cuotas_ant, v_reac_prev, v_clv, v_saldo, v_persev])

    # D5 · EXTERNO
    v_alerta_habito     = 1 - df["Alerta_Habito_Pago_Externo"]
    v_alerta_creditos   = 1 - df["Alerta_Estado_Creditos_Externos"]
    v_alerta_capacidad  = 1 - df["Alerta_Capacidad_Pago_Externo"]
    d5 = score_dimension([v_alerta_habito, v_alerta_creditos, v_alerta_capacidad])

    df["d1_severidad"] = d1
    df["d2_enganche"]  = d2
    df["d3_recencia"]  = d3
    df["d4_vinculo"]   = d4
    df["d5_externo"]   = d5

    df["score_compromiso"] = (
        d1 * pesos["severidad"] +
        d2 * pesos["enganche"] +
        d3 * pesos["recencia"] +
        d4 * pesos["vinculo"] +
        d5 * pesos["externo"]
    ) * 100

    return df[["Periodo", "Id",
               "d1_severidad", "d2_enganche", "d3_recencia", "d4_vinculo", "d5_externo",
               "score_compromiso"]]


def categorizar_score(score: pd.Series) -> pd.Series:
    """Asigna categoría Bajo/Medio/Alto al score usando media ± 0.5·std como cortes.

    Args:
        score (pd.Series): Serie numérica con el score_compromiso en [0, 100].

    Returns:
        pd.Series: Serie categórica con etiquetas ['Bajo', 'Medio', 'Alto'].

    Note:
        Persiste ``configs/scoring/cortes_scoring.json`` como efecto secundario.
    """
    media = score.mean()
    std   = score.std()

    corte_bajo = round(media - 0.5 * std, 0)
    corte_alto = round(media + 0.5 * std, 0)

    particiones = {
        "bajo":  [0,          corte_bajo],
        "medio": [corte_bajo, corte_alto],
        "alto":  [corte_alto, 100],
    }

    path_out = Path("configs/scoring")
    path_out.mkdir(parents=True, exist_ok=True)
    with open(path_out / "cortes_scoring.json", "w", encoding="utf-8") as f:
        json.dump(particiones, f)
    print(f"Exportado: {path_out / 'cortes_scoring.json'}")

    return pd.cut(
        score,
        bins=[0, corte_bajo, corte_alto, 100],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    )

# ─── Orquestadora ─────────────────────────────────────────────────────────────

def build_score(periodo: str | None = None) -> None:
    """Orquesta el pipeline de scoring D1-D5 desde la base analítica hasta scoring.parquet.

    Flujo de ejecución:
        1. Crea ``models/score/`` si no existe.
        2. Carga orden_clv y pesos (severidad/enganche/recencia/vinculo/externo)
           desde config.yml -> scoring (ya suman 1.0).
        3. Lee las columnas necesarias de
           ``data/analytic/{periodo}/analytic_score_base.parquet`` (el periodo
           más reciente si no se especifica).
        4. Llama a `calcular_scoring` -> genera dimensiones D1-D5 y score_compromiso.
        5. Llama a `categorizar_score` -> genera categoria_score y persiste cortes.
        6. Exporta el DataFrame final a ``data/scoring/scoring_inactivos.parquet``.

    Args:
        periodo: Periodo (YYYYMM) a puntuar, ej. '202608'. Si es None, se toma
            la corrida de transformación más reciente en ``data/analytic/``.

    Raises:
        FileNotFoundError: Si la base analítica o config.yml no existen.
        KeyError: Si config.yml no contiene las claves esperadas bajo 'scoring'.
    """
    Path("models/score").mkdir(parents=True, exist_ok=True)

    print("Importando dependencias scoring...")
    cfg = load_config()
    orden_clv = cfg["scoring"]["orden_clv"]
    pesos = cfg["scoring"]

    df_scoring = xtr_columnas_necesarias(periodo=periodo)

    score = calcular_scoring(
        df=df_scoring,
        orden_clv=orden_clv,
        pesos=pesos,
    )

    score["categoria_score"] = categorizar_score(score["score_compromiso"])

    path_out = Path("data/scoring")
    path_out.mkdir(parents=True, exist_ok=True)
    score.to_parquet(path_out / "scoring_inactivosV2.parquet", index=False, engine="pyarrow")
    print(f"Exportado: {path_out / 'scoring_inactivosv2.parquet'}")
