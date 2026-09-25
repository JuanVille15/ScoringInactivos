import os
from dotenv import load_dotenv
from src.extraccion.extract_raw import extract_raw
from src.transformacion.build_analytic_score import build_analytic_score
from src.Scoring.build_score import build_score
from src.Scoring.caracterizar_score import caracterizar_score
from src.Scoring.zoom_alta import build_zoom
from src.utils.helpers import version_scoring, path_artefactos, path_cortes, path_entrenamiento

import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    load_dotenv()
    con_bi = os.environ["CON_BI"]
    con_gcc = os.environ["GCC_CON"]

    # --- 0. VERSIÓN: entrena la versión de config.yml -> scoring.version. No
    # pisa una versión que ya existe (inference.py la puede estar usando): para
    # reentrenar, sube la versión en config.yml (ej. v2 -> v3). ---
    version = version_scoring()
    if path_cortes(version).exists() or any(path_artefactos(version).glob("*.pkl")):
        raise FileExistsError(
            f"La versión {version} ya está entrenada ({path_cortes(version)} / "
            f"{path_artefactos(version)}). Sube scoring.version en config.yml para "
            "entrenar una nueva, o borra esos artefactos si de verdad quieres rehacerla."
        )

    # Todo lo que produce el entrenamiento va a data/train/{version}/, no a
    # data/raw, data/analytic ni data/scoring (esas son de inferencia).
    raiz_train = path_entrenamiento(version)
    print(f"Entrenando score {version} -> {raiz_train}")

    # --- 1. EXTRACCIÓN: trae todas las bases crudas necesarias para el proceso ---
    bases = extract_raw(con_bi=con_bi, con_gcc=con_gcc, raw_path=str(raiz_train / "raw"))

    # --- 2. TRANSFORMACIÓN: arma la base analítica de scoring D1-D5 (severidad,
    # enganche, recencia, vínculo, externo) a partir de `bases` ---
    analytic = build_analytic_score(bases=bases, analytic_path=str(raiz_train / "analytic"))
    periodo = str(analytic["Periodo"].iloc[0])
    path_analytic = str(raiz_train / "analytic" / periodo / "analytic_score_base.parquet")

    # --- 3. SCORING: entrena escaladores + cortes de la versión ---
    build_score(analytic_path=path_analytic)

    # --- 4. CARACTERIZACIÓN: valida la distribución del score y resume las
    # variables crudas por grupo Bajo/Medio/Alto (diagnóstico) ---
    caracterizar_score(analytic_path=path_analytic)

    # --- 5. Se Genera priorizacion para alta --- #
    build_zoom(analytic_path=path_analytic)

if __name__ == "__main__":
    main()
