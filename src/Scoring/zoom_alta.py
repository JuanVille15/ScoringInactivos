"""Zoom - Mora alta

Este modulo realiza un "mini-scoring" para generar una priorizacion numerica
entre 5 variables - Oferta Disponible, NumeroProductos, Cuotas pagadas vs antiguedad,
Saldo aportes y ValorCapitalizado posterior a eso organiza de mayor a menor por el score y genera particiones. 
"""
import joblib
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
from src.Scoring.build_score import clampear_percentil

def xtr_bases(analytic_path: str | None = None, 
             scoring_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # --- Se extrae la base analitica --- #
         
    path_base = Path(analytic_path) if analytic_path else Path().cwd() /"data"/ "analytic" / "analytic_score_base.parquet"
    
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
    
    path_scoring = Path(scoring_path) if scoring_path else Path().cwd() /"data"/ "scoring" / "scoring_inactivosV2.parquet"
    
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
    
    ARTIFACTS_BASE_PATH = Path().cwd() / "models" / "score"
    
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

def agruparzoom(df:pd.DataFrame) -> pd.DataFrame:
    
    # --- Se genera particiones por cuantiles --- #
    
    df['PriorizacionNumerica'] = (
        pd.qcut(
            x= df['ZoomAlta'], 
            q=3, 
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

def build_zoom() -> None:
    
    # --- 1. Se extraen bases necesarias --- #
    features, alta = xtr_bases()
    
    # --- 2. Se unen las bases ---- #
    FeaturesAlta = unir_bases(
        features=features, 
        alta=alta, 
    )
    
    # --- 3. Se genera la columna zoom --- #
    FeaturesAlta = zoom(
        df=FeaturesAlta, 
    )
    
    # --- 4. Se genera priorizacion numerica --- #
    FeaturesAlta = agruparzoom(
        df=FeaturesAlta, 
    )
    
    # ==========
    # EXPORTAR
    # ==========
    path_out = Path("data/scoring")
    path_out.mkdir(parents=True, exist_ok=True)
    print(f"Exportado: {path_out / 'priorizacion_altaV2.xlsx'}...")
    FeaturesAlta.to_excel(path_out / "priorizacion_alta.xlsx", index=False,)
    print('Priorizacion Correctamente Exportada ✅')
