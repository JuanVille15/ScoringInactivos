"""Zoom - Mora alta

Este modulo realiza un "mini-scoring" para generar una priorizacion numerica
entre 4 variables - NumeroProductos, Cuotas pagadas vs antiguedad,
Saldo aportes y ValorCapitalizado posterior a eso organiza de mayor a menor por el score y genera particiones.
"""
import joblib
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
from src.Scoring.build_score import clampear_percentil
from src.utils.helpers import (
    periodo_mas_cercano, path_artefactos, path_entrenamiento, path_cortes_zoom, leer_cortes_zoom,
)

def xtr_bases(periodo: str | None = None,
             analytic_path: str | None = None,
             scoring_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- Se extrae la base analitica --- #

    if analytic_path is not None:
        path_base = Path(analytic_path)
    else:
        raiz_analytic = Path().cwd() / "data" / "analytic"
        periodo_resuelto = periodo or periodo_mas_cercano(raiz_analytic, "analytic_score_base.parquet")
        path_base = raiz_analytic / periodo_resuelto / "analytic_score_base.parquet"

    if not path_base.exists():
        raise FileNotFoundError(f'No existe {path_base}. Corre primero el modulo de transformacion')
    
    COL_NECESARIAS = [
        "Id",
        "Numcantidadproductos", 
        "Cuotas_pagadas_vs_antiguedad", 
        "Saldoaportes", 
        "Valor_Capitalizado", 
    ]
    
    features = (pd.read_parquet(path_base, 
                             engine='pyarrow', 
                             )
                .loc[:,COL_NECESARIAS])
    
    # --- Se extrae la base de altos --- #
    
    path_scoring = Path(scoring_path) if scoring_path else path_entrenamiento() / "scoring_inactivos.parquet"
    
    alta = (
        pd.read_parquet(
            path_scoring, 
            engine='pyarrow', 
        )
        .loc[
            lambda df:
                df['categoria_score'] == 'Alto', 
                :
        ]
    )

    return features, alta

def unir_bases(features:pd.DataFrame, 
               alta:pd.DataFrame) -> pd.DataFrame:
    
    
    # --- Se unen las dos bases --- #
    
    FeaturesAlta = (
        pd.merge(
            left=alta[['Id']]
            .astype(int), 
            right=features
            .assign(
                Id = lambda df:
                    df['Id'].astype(int), 
            ), 
            how='left', 
            on='Id', 
        )
    )

    return FeaturesAlta

def inferencia_continua(series: pd.Series, 
                        nombre: str, 
                        invertir: bool = False, 
                        kind: Literal['Continua','log'] = 'Continua') -> pd.Series:
    
    # --- Se camplea la variable --- # 
    
    if kind == 'log':
        s = pd.Series(np.log1p(clampear_percentil(series)))
    else:
        s = clampear_percentil(series)
        
    # --- Se importa el artifact --- #
    
    ARTIFACTS_BASE_PATH = path_artefactos()
    
    if not ARTIFACTS_BASE_PATH.exists():
        raise FileExistsError(f'La carpeta de artefactos no existe: {ARTIFACTS_BASE_PATH}')
    
    artifact = sorted(ARTIFACTS_BASE_PATH.glob(f"*{nombre}.pkl"), key= lambda a: os.path.getmtime(a), reverse=True)[0]
    
    if not artifact.exists():
        raise FileExistsError(f'No existe artefacto entrenado: {nombre}')
    else:
        print(f'Cargando Artefacto: {artifact}...')
        
    try:
        with open(artifact, 'rb') as file:
            scaler = joblib.load(file)
    except Exception as e:
        print(f'Error Cargando Artefacto: {artifact} - {e}')
        raise e
    
    # --- Se genera la inferencia --- #
    try:
        serie_normalizada = scaler.transform(s.to_numpy().reshape(-1,1)).flatten()
        serie_salida = pd.Series(data=serie_normalizada, 
                                 index=s.index)
    except Exception as a:
        print(f'Ocurrio un error normalizando: {nombre} - {a}')
        raise a
    
    return 1 - serie_salida if invertir else serie_salida

def zoom(
    df: pd.DataFrame, 
) -> pd.DataFrame:
    
    # ==============
    # Imputaciones
    # ==============
    df = df.copy()
    df["Numcantidadproductos"] = df["Numcantidadproductos"].fillna(0)
    
    # ===========================
    # Normalizar las variables
    # ===========================
    
    # --- 1. Numcantidadproductos --- #
    NumCantidadProductos = inferencia_continua(
        series=df['Numcantidadproductos'], 
        nombre='num_productos', 
        invertir=False, 
        kind='Continua', 
    )
    
    # --- 2. Cuotas_pagadas_vs_antiguedad --- #
    CuotasAntiguedad = inferencia_continua(
        series=df['Cuotas_pagadas_vs_antiguedad'], 
        nombre='cuotas_ant', 
        invertir=False, 
        kind='Continua', 
    )
    
    # --- 3. Saldoaportes --- #
    SaldoAportes = inferencia_continua(
        series=df['Saldoaportes'], 
        nombre='saldo', 
        invertir=False, 
        kind='log', 
    )
    
    # --- 4. ValorCapitalizado --- #
    ValorCapitalizado = inferencia_continua(
        series=df['Valor_Capitalizado'], 
        nombre='valor_capitalizado', 
        invertir=False, 
        kind='log', 
    )
    
    # =======================
    # Crear la serie de ZOOM
    # =======================
    
    zoom = pd.concat([NumCantidadProductos, CuotasAntiguedad, 
                      SaldoAportes, ValorCapitalizado], 
                    axis=1).mean(axis=1, skipna=True)

    # =========================
    # Etiquetar el df original
    # =========================
    
    df['ZoomAlta'] = zoom
    
    return df

def entrenar_cortes_zoom(zoom_alta: pd.Series) -> list[float]:
    """Calcula los terciles de ZoomAlta sobre las cédulas Alto del lote de
    entrenamiento y los persiste en ``configs/scoring/cortes_zoom_{version}.json``.

    Misma metodología que los cortes del score: se calculan una sola vez al
    entrenar y en inferencia se aplican congelados, para que la prioridad
    1/2/3 de una cédula sea comparable entre corridas (antes `pd.qcut` se
    recalculaba en cada corrida y era relativo a ese lote).

    Args:
        zoom_alta: Serie ZoomAlta de las cédulas Alto del entrenamiento.

    Returns:
        Los 2 cortes interiores [tercil_1, tercil_2].
    """
    _, bins = pd.qcut(x=zoom_alta, q=3, retbins=True)
    cortes = [float(bins[1]), float(bins[2])]

    path_json = path_cortes_zoom()
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"cortes": cortes}, f)
    print(f"Exportado: {path_json} -- {cortes}")

    return cortes


def agruparzoom(df:pd.DataFrame, cortes: list[float] | None = None) -> pd.DataFrame:
    """Asigna PriorizacionNumerica (1 = más prioritario) con cortes congelados.

    Args:
        df: DataFrame con la columna 'ZoomAlta'.
        cortes: [tercil_1, tercil_2]. Si es None, se leen los de la versión
            vigente (`leer_cortes_zoom`) — el caso de inferencia.
    """
    cortes = cortes if cortes is not None else leer_cortes_zoom()

    # --- Se particiona con los terciles del entrenamiento --- #
    # Intervalos cerrados a la derecha, igual que pd.qcut.
    df['PriorizacionNumerica'] = (
        pd.cut(
            x= df['ZoomAlta'],
            bins=[-np.inf, cortes[0], cortes[1], np.inf],
            labels=[3,2,1],
        )
    )
    
    # --- Se organiza el dataframe por priorizacion numerica --- #
    
    df = (
        df
        .sort_values(
            by='PriorizacionNumerica', 
            ascending=False, 
            kind='quicksort', 
            ignore_index=True, 
        )
    )

    return df

def build_zoom(periodo: str | None = None, analytic_path: str | None = None) -> None:

    # --- 1. Se extraen bases necesarias --- #
    features, alta = xtr_bases(periodo=periodo, analytic_path=analytic_path)
    
    # --- 2. Se unen las bases ---- #
    FeaturesAlta = unir_bases(
        features=features, 
        alta=alta, 
    )
    
    # --- 3. Se genera la columna zoom --- #
    FeaturesAlta = zoom(
        df=FeaturesAlta, 
    )
    
    # --- 4. Se entrenan los terciles y se genera priorizacion numerica --- #
    cortes = entrenar_cortes_zoom(FeaturesAlta['ZoomAlta'])
    FeaturesAlta = agruparzoom(
        df=FeaturesAlta,
        cortes=cortes,
    )
    
    # ==========
    # EXPORTAR
    # ==========
    path_out = path_entrenamiento()
    path_out.mkdir(parents=True, exist_ok=True)
    print(f"Exportado: {path_out / 'priorizacion_alta.xlsx'}...")
    FeaturesAlta.to_excel(path_out / "priorizacion_alta.xlsx", index=False,)
    print('Priorizacion Correctamente Exportada')
