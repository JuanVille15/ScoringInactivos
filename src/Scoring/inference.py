"""Inferencia D1-D5 para la población de inactivos.

A diferencia de `build_score.py` (entrenamiento, llamado desde train.py), acá
no se reajusta nada: cada variable pasa por `.transform()` sobre el escalador
ya entrenado en `models/score/` (no `.fit_transform()`), y la categoría
Bajo/Medio/Alto usa los cortes ya congelados en
`configs/scoring/cortes_scoring.json` (no se recalculan). Así el score de una
cédula es comparable entre corridas de inferencia distintas — ver la nota de
CLAUDE.md sobre por qué `build_score.py` no sirve tal cual para esto.

Además de puntuar, separa qué cédulas de la corrida ya estaban etiquetadas en
el maestro histórico (`data/scoring/scoring_inactivos.parquet`) de las
genuinamente nuevas, y prioriza numéricamente a las cédulas categoría Alto
(reusando `zoom_alta.py`, que también es transform-only).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.utils.config import load_config
from src.utils.helpers import leer_cortes_scoring
from src.Scoring.build_score import clampear_percentil, score_dimension
from src.Scoring.zoom_alta import unir_bases, zoom, agruparzoom

# ─── Carga de artefactos ya entrenados ─────────────────────────────────────────

def _cargar_artefacto(nombre: str, prefijo: str = "MinMax"):
    """Carga un artefacto ya entrenado (escalador u encoder) de models/score/.

    Args:
        nombre: Nombre de la variable (ej. 'saldo' -> MinMax_saldo.pkl).
        prefijo: 'MinMax' (default) u 'OrdinalEncoder'.

    Returns:
        El objeto sklearn ya ajustado (MinMaxScaler u OrdinalEncoder).

    Raises:
        FileNotFoundError: Si el artefacto no existe — inference.py no
            entrena nada, así que hay que correr train.py (build_score) al
            menos una vez antes.
    """
    path = Path("models/score") / f"{prefijo}_{nombre}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el artefacto entrenado {path}. Corre primero train.py "
            "(build_score) al menos una vez para generarlo."
        )
    return joblib.load(path)


# ─── Normalización en modo inferencia (transform, no fit_transform) ───────────
# Réplica exacta de normalizar_continua/log/zero_inflated/categoricas en
# build_score.py, cambiando fit_transform+dump por cargar+transform.

def inferir_continua(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Ver `normalizar_continua` en build_score.py — misma lógica, sin reajustar."""
    s = clampear_percentil(series)
    scaler = _cargar_artefacto(name)
    s_scaled = scaler.transform(s.to_numpy().reshape(-1, 1)).flatten()
    serie_salida = pd.Series(data=s_scaled, index=s.index)
    return 1 - serie_salida if invertir else serie_salida


def inferir_log(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Ver `normalizar_log` en build_score.py — misma lógica, sin reajustar."""
    s = np.log1p(clampear_percentil(series))
    scaler = _cargar_artefacto(name)
    s_scaled = scaler.transform(s.to_numpy().reshape(-1, 1)).flatten()  # type: ignore
    serie_salida = pd.Series(data=s_scaled, index=series.index)
    return 1 - serie_salida if invertir else serie_salida


def inferir_zero_inflated(series: pd.Series, name: str, invertir: bool = False) -> pd.Series:
    """Ver `normalizar_zero_inflated` en build_score.py — misma lógica, sin reajustar."""
    s = series.fillna(0).copy()

    mask_pos = s > 0
    s_norm = pd.Series(0.0, index=s.index)

    if mask_pos.sum() > 0:
        s_pos = np.log1p(clampear_percentil(s[mask_pos]))
        scaler = _cargar_artefacto(name)
        s_pos_scaled = scaler.transform(s_pos.to_numpy().reshape(-1, 1)).flatten()  # type: ignore
        s_norm[mask_pos] = 0.1 + s_pos_scaled * 0.9  # type: ignore[operator]

    return 1 - s_norm if invertir else s_norm


def inferir_categoricas(series: pd.Series, name: str) -> pd.Series:
    """Ver `normalizar_categoricas` en build_score.py — misma lógica, sin reajustar."""
    s = series.copy()

    enc = _cargar_artefacto(name, prefijo="OrdinalEncoder")
    serie_codificada = enc.transform(s.to_numpy().reshape(-1, 1)).flatten()

    scaler = _cargar_artefacto(name, prefijo="MinMax")
    serie_normalizada = scaler.transform(serie_codificada.reshape(-1, 1)).flatten()

    return pd.Series(serie_normalizada, index=s.index)


# ─── Pipeline de scoring en inferencia ─────────────────────────────────────────

def inferir_scoring(df: pd.DataFrame, orden_clv: list, pesos: dict) -> pd.DataFrame:
    """Igual que `calcular_scoring` (build_score.py) pero sin reentrenar.

    Mismo agrupamiento de dimensiones, mismos pesos, misma fórmula final —
    ver `calcular_scoring` para el detalle de cada dimensión. La única
    diferencia es que cada variable pasa por `inferir_*` (transform sobre el
    artefacto ya entrenado) en vez de `normalizar_*` (fit_transform + guardar).

    Args:
        df: DataFrame con Periodo, Id y las variables de D1-D5 (ej. la salida
            de `build_analytic_score`).
        orden_clv: Categorías de Clv ordenadas de menor a mayor valor
            (config.yml -> scoring.orden_clv). No se usa para reajustar el
            encoder (ya está entrenado), solo la recibe `calcular_scoring`
            para tener la misma firma — se ignora en la práctica.
        pesos: Pesos con claves 'severidad', 'enganche', 'recencia', 'vinculo',
            'externo' (config.yml -> scoring).

    Returns:
        DataFrame con columnas ['Periodo', 'Id', 'd1_severidad', 'd2_enganche',
        'd3_recencia', 'd4_vinculo', 'd5_externo', 'score_compromiso'].

    Raises:
        FileNotFoundError: Si falta algún artefacto en models/score/.
    """
    df = df.copy()

    df["Numcantidadproductos"] = df["Numcantidadproductos"].fillna(0)
    df["Cantidad_empresas"]    = df["Cantidad_empresas"].fillna(0)

    # D1 · SEVERIDAD INACTIVIDAD
    v_tiempo_inact  = inferir_continua(df["Tiempo_Inactividad_Meses"],              name="tiempo_inactividad", invertir=True)
    v_saldo_deuda   = inferir_continua(df["Pct_Mora_GECC_6M"],                      name="saldo_deuda",       invertir=True)
    v_pagos_parc    = inferir_zero_inflated(df["Recaudo_Total_Promedio_GECC_6M"],    name="pagos_parciales")
    v_intenc_retiro = inferir_zero_inflated(df["intenciones_retiro_1y"],            name="intencion_retiro",  invertir=True)
    d1 = score_dimension([v_tiempo_inact, v_saldo_deuda, v_pagos_parc, v_intenc_retiro])

    # D2 · ENGANCHE
    v_numprod    = inferir_continua(df["Numcantidadproductos"],                    name="num_productos")
    v_empresas   = inferir_continua(df["Cantidad_empresas"],                       name="num_empresas")
    v_proteccion = inferir_log(df["Valor_perseverancia_vs_ingresos"],              name="proteccion")
    v_n_usos     = inferir_log(df["Total_Eventos"],                                name="n_usos")
    v_valor_cap  = inferir_log(df["Valor_Capitalizado"],                           name="valor_capitalizado")
    d2 = score_dimension([v_numprod, v_empresas, v_proteccion, v_n_usos, v_valor_cap])

    # D3 · RECENCIA
    v_dist_reac    = inferir_log(df["Distancia_ultima_reactivacion"],              name="dist_reac")
    v_rec_prod     = inferir_continua(df["Recencia_ultimo_producto"],              name="rec_prod",      invertir=True)
    v_turnos_ofic  = inferir_zero_inflated(df["Turnos_En_Oficinas_Total_Ult12Meses"], name="turnos_oficina")
    v_pqr          = inferir_zero_inflated(df["Cantidad_pqr_ultimo_anno"],          name="pqr",            invertir=True)
    d3 = score_dimension([v_dist_reac, v_rec_prod, v_turnos_ofic, v_pqr])

    # D4 · VÍNCULO HISTÓRICO
    v_cuotas_ant = inferir_continua(df["Cuotas_pagadas_vs_antiguedad"],            name="cuotas_ant")
    v_reac_prev  = inferir_continua(df["Cantidad_Reactivaciones_Previas"],         name="reac_prev",     invertir=True)
    v_clv        = inferir_categoricas(df["Clv"],                                  name="clv")
    v_saldo      = inferir_log(df["Saldoaportes"],                                 name="saldo")
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


def etiquetar_categoria(score: pd.Series, cortes: dict) -> pd.Series:
    """Asigna categoría Bajo/Medio/Alto usando los cortes ya congelados.

    A diferencia de `categorizar_score` (build_score.py), no calcula ni
    persiste cortes nuevos: usa los que ya existen en
    configs/scoring/cortes_scoring.json (vía `leer_cortes_scoring`), para que
    la categoría de una cédula sea comparable entre corridas de inferencia.

    Args:
        score: Serie con score_compromiso en [0, 100].
        cortes: dict con claves 'bajo', 'medio', 'alto' (ver `leer_cortes_scoring`).

    Returns:
        Serie categórica con etiquetas ['Bajo', 'Medio', 'Alto'].
    """
    corte_bajo = cortes["bajo"][1]
    corte_alto = cortes["alto"][0]

    return pd.cut(
        score,
        bins=[0, corte_bajo, corte_alto, 100],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    )


# ─── Split nuevos / ya etiquetados ─────────────────────────────────────────────

def separar_nuevos(score: pd.DataFrame, path_maestro: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la población puntuada esta corrida entre 'ya etiquetada antes' y 'nueva'.

    'Ya etiquetada' se determina solo por Id (no por Id+Periodo): una cédula
    que sigue inactiva mes a mes y ya se etiquetó una vez no vuelve a contar
    como nueva, aunque su Periodo cambie en cada corrida — así es como se
    evita reprocesarla en la corrida del mes siguiente.

    Args:
        score: Población completa puntuada en esta corrida (salida de
            `inferir_scoring` + `etiquetar_categoria`).
        path_maestro: Ruta al parquet maestro histórico
            (data/scoring/scoring_inactivos.parquet). Si no existe todavía
            (primera corrida), se trata como vacío: toda `score` es "nueva".

    Returns:
        (nuevos, maestro_actualizado): `nuevos` es el subconjunto de `score`
        con Id no presente en el maestro; `maestro_actualizado` es el maestro
        original con `nuevos` ya appendeado (pd.concat), listo para persistir
        de vuelta en `path_maestro`.
    """
    if path_maestro.exists():
        maestro = pd.read_parquet(path_maestro, engine="pyarrow")
        ids_ya_etiquetados = set(maestro["Id"].astype(str))
    else:
        maestro = pd.DataFrame(columns=score.columns)
        ids_ya_etiquetados = set()

    nuevos = score[~score["Id"].astype(str).isin(ids_ya_etiquetados)].copy()
    maestro_actualizado = pd.concat([maestro, nuevos], ignore_index=True)

    return nuevos, maestro_actualizado


# ─── Priorización (categoría Alto) ─────────────────────────────────────────────

def priorizar_alta(score: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    """Prioriza numéricamente a las cédulas categoría Alto de esta corrida.

    Reusa zoom_alta.py tal cual: `zoom()` ahí adentro ya es transform-only
    (`inferencia_continua` carga y aplica los artefactos ya entrenados, no
    reentrena), así que no hay nada que adaptar. Se aplica a TODAS las
    cédulas Alto de esta corrida (no solo a las nuevas): la priorización es
    "a quién contactar ya", independiente de si la cédula ya se había
    etiquetado en una corrida anterior.

    Args:
        score: Población completa puntuada en esta corrida, con categoria_score.
        analytic: DataFrame de `build_analytic_score` para esta misma corrida
            (trae las 4 variables que usa zoom: Numcantidadproductos,
            Cuotas_pagadas_vs_antiguedad, Saldoaportes, Valor_Capitalizado).

    Returns:
        DataFrame con las cédulas Alto, sus 4 variables, 'ZoomAlta' y
        'PriorizacionNumerica' (1 = más prioritario). Vacío (mismas columnas,
        0 filas) si esta corrida no tiene ninguna cédula Alto.
    """
    alta = score.loc[score["categoria_score"] == "Alto", ["Id"]]

    if alta.empty:
        print("[inference] No hay cédulas categoría Alto en esta corrida -- no se genera priorización.")
        return pd.DataFrame(columns=[
            "Id", "Numcantidadproductos", "Cuotas_pagadas_vs_antiguedad",
            "Saldoaportes", "Valor_Capitalizado", "ZoomAlta", "PriorizacionNumerica",
        ])

    features = analytic[[
        "Id", "Numcantidadproductos", "Cuotas_pagadas_vs_antiguedad",
        "Saldoaportes", "Valor_Capitalizado",
    ]]

    df = unir_bases(features=features, alta=alta)
    df = zoom(df=df)
    df = agruparzoom(df=df)

    return df


# ─── Exportación ────────────────────────────────────────────────────────────────

def exportar_inferencia(
    score: pd.DataFrame,
    nuevos: pd.DataFrame,
    maestro_actualizado: pd.DataFrame,
    priorizacion: pd.DataFrame,
    periodo: str,
    path_maestro: Path,
    path_out_compartido: str,
    scoring_root: str | None = None,
) -> None:
    """Persiste todos los artefactos de salida de una corrida de inferencia.

    Deja en ``{scoring_root}/{periodo}/``:
        - scoring_inactivos.parquet: TODA la población puntuada esta corrida.
        - scoring_nuevos_inactivos.parquet: solo las cédulas nuevas.
        - priorizacion_numerica_alta.xlsx: priorización de las cédulas Alto.

    Sobreescribe el maestro histórico (`path_maestro`, normalmente
    ``data/scoring/scoring_inactivos.parquet``) con `maestro_actualizado`
    (ya incluye las nuevas de esta corrida) — así una cédula ya etiquetada no
    vuelve a salir como "nueva" en una corrida futura.

    En la ruta compartida (config.yml -> paths.out_score), bajo
    ``{periodo}/``, deja ``score_nuevos_inactivos.xlsx`` (mismo contenido que
    el parquet de nuevos) y otra copia de ``priorizacion_numerica_alta.xlsx``.
    Si `path_out_compartido` está vacía, se omiten estas 2 (config.yml aún
    sin la ruta de red configurada) — las exportaciones locales sí se hacen
    siempre.

    Args:
        score: Población completa puntuada esta corrida.
        nuevos: Subconjunto de `score` con cédulas no vistas antes.
        maestro_actualizado: Maestro histórico + `nuevos` ya appendeado.
        priorizacion: Salida de `priorizar_alta`.
        periodo: Periodo (YYYYMM) de esta corrida.
        path_maestro: Ruta al parquet maestro (normalmente
            data/scoring/scoring_inactivos.parquet).
        path_out_compartido: Ruta de red compartida (config.yml -> paths.out_score).
        scoring_root: Carpeta raíz local personalizada. Si es None, usa
            ``data/scoring``.
    """
    # --- 1. {scoring_root}/{periodo}/: población completa + solo nuevas ---
    path_periodo = Path(scoring_root) / periodo if scoring_root else Path("data/scoring") / periodo
    path_periodo.mkdir(parents=True, exist_ok=True)

    score.to_parquet(path_periodo / "scoring_inactivos.parquet", index=False, engine="pyarrow")
    print(f"[inference] Exportado: {path_periodo / 'scoring_inactivos.parquet'} -- {len(score):,} cédulas (total de la corrida)")

    nuevos.to_parquet(path_periodo / "scoring_nuevos_inactivos.parquet", index=False, engine="pyarrow")
    print(f"[inference] Exportado: {path_periodo / 'scoring_nuevos_inactivos.parquet'} -- {len(nuevos):,} cédulas (nuevas)")

    # --- 2. Maestro histórico actualizado (in-place) ---
    path_maestro.parent.mkdir(parents=True, exist_ok=True)
    maestro_actualizado.to_parquet(path_maestro, index=False, engine="pyarrow")
    print(f"[inference] Maestro actualizado: {path_maestro} -- {len(maestro_actualizado):,} cédulas acumuladas")

    # --- 3. Priorización numérica Alto: copia local ---
    priorizacion.to_excel(path_periodo / "priorizacion_numerica_alta.xlsx", index=False)
    print(f"[inference] Exportado: {path_periodo / 'priorizacion_numerica_alta.xlsx'} -- {len(priorizacion):,} cédulas Alto")

    # --- 4. Ruta compartida: nuevos .xlsx + copia de priorización ---
    if not path_out_compartido:
        print(
            "[inference] paths.out_score está vacío en config.yml: se omite la "
            "exportación a la ruta compartida (score_nuevos_inactivos.xlsx y "
            "priorizacion_numerica_alta.xlsx)."
        )
        return

    path_out_periodo = Path(path_out_compartido) / periodo
    path_out_periodo.mkdir(parents=True, exist_ok=True)

    nuevos.to_excel(path_out_periodo / "score_nuevos_inactivos.xlsx", index=False)
    print(f"[inference] Exportado (compartido): {path_out_periodo / 'score_nuevos_inactivos.xlsx'} -- {len(nuevos):,} cédulas")

    priorizacion.to_excel(path_out_periodo / "priorizacion_numerica_alta.xlsx", index=False)
    print(f"[inference] Exportado (compartido): {path_out_periodo / 'priorizacion_numerica_alta.xlsx'} -- {len(priorizacion):,} cédulas Alto")


# ─── Orquestadora ─────────────────────────────────────────────────────────────

def ejecutar_inferencia(
    analytic: pd.DataFrame,
    periodo: str,
    scoring_root: str | None = None,
) -> None:
    """Orquesta la inferencia completa (scoring + priorización) para una
    corrida ya extraída y transformada.

    Función de entrada principal del módulo. A diferencia de
    build_score()+build_zoom() (entrenamiento), no reajusta nada: usa los
    artefactos ya entrenados en models/score/ y los cortes Bajo/Medio/Alto ya
    congelados en configs/scoring/cortes_scoring.json.

    Args:
        analytic: DataFrame ya armado por `build_analytic_score` (todas las
            variables D1-D5 + Periodo/Id) para el periodo que se va a puntuar.
        periodo: Periodo (YYYYMM) de esta corrida, ej. '202608'. Nombra las
            carpetas de salida.
        scoring_root: Carpeta raíz local personalizada para la salida y el
            maestro histórico. Si es None, usa ``data/scoring``.

    Raises:
        FileNotFoundError: Si falta algún artefacto en models/score/ o el
            json de cortes (correr primero train.py).
    """
    print(f"[inference] Puntuando periodo {periodo} -- {len(analytic):,} cédulas...")

    cfg = load_config()
    orden_clv = cfg["scoring"]["orden_clv"]
    pesos = cfg["scoring"]

    score = inferir_scoring(analytic, orden_clv, pesos)

    cortes = leer_cortes_scoring()
    score["categoria_score"] = etiquetar_categoria(score["score_compromiso"], cortes)

    raiz_scoring = Path(scoring_root) if scoring_root else Path("data/scoring")
    path_maestro = raiz_scoring / "scoring_inactivos.parquet"
    nuevos, maestro_actualizado = separar_nuevos(score, path_maestro)

    print(f"[inference] Total puntuadas esta corrida: {len(score):,}")
    print(f"[inference] Ya etiquetadas antes (se excluyen de 'nuevos'): {len(score) - len(nuevos):,}")
    print(f"[inference] Nuevas cédulas: {len(nuevos):,}")

    priorizacion = priorizar_alta(score=score, analytic=analytic)

    exportar_inferencia(
        score=score,
        nuevos=nuevos,
        maestro_actualizado=maestro_actualizado,
        priorizacion=priorizacion,
        periodo=periodo,
        path_maestro=path_maestro,
        path_out_compartido=cfg["paths"].get("out_score", "") or "",
        scoring_root=scoring_root,
    )
