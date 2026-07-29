import os
from dotenv import load_dotenv
from src.extraccion.extract_raw import extract_raw
from src.transformacion.build_analytic_score import build_analytic_score
from src.Scoring.build_score import build_score
from src.Scoring.caracterizar_score import caracterizar_score
from src.Scoring.zoom_alta import build_zoom

import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    load_dotenv()
    con_bi = os.environ["CON_BI"]

    # --- 1. EXTRACCIÓN: trae todas las bases crudas necesarias para el proceso ---
    bases = extract_raw(con_bi=con_bi)

    # --- 2. TRANSFORMACIÓN: arma la base analítica de scoring D1-D5 (severidad,
    # enganche, recencia, vínculo, externo) a partir de `bases` ---
    build_analytic_score(bases=bases)

    # --- 3. SCORING: aplica el modelo de scoring D1-D5 sobre la base analítica ---
    build_score()

    # --- 4. CARACTERIZACIÓN: valida la distribución del score y resume las
    # variables crudas por grupo Bajo/Medio/Alto (diagnóstico) ---
    caracterizar_score()
    
    # --- 5. Se Genera priorizacion para alta --- #
    build_zoom()
    
if __name__ == "__main__":
    main()