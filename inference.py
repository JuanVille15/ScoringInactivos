import os
from dotenv import load_dotenv
from src.extraccion.extract_raw import extract_raw
from src.transformacion.build_analytic_score import build_analytic_score
from src.Scoring.inference import ejecutar_inferencia

import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    load_dotenv()
    con_bi = os.environ["CON_BI"]
    con_gcc = os.environ["GCC_CON"]

    print("=" * 70)
    print("INFERENCE.PY -- Inferencia de scoring (D1-D5) para inactivos")
    print("Sin reentrenar: usa los artefactos ya entrenados en models/score/")
    print("=" * 70)

    # --- 1. EXTRACCIÓN: trae todas las bases crudas necesarias para el proceso ---
    print("\n[1/3] Extrayendo bases crudas...")
    bases = extract_raw(con_bi=con_bi, con_gcc=con_gcc)

    # --- 2. TRANSFORMACIÓN: arma la base analítica de scoring D1-D5 (severidad,
    # enganche, recencia, vínculo, externo) a partir de `bases` ---
    print("\n[2/3] Transformando -- armando base analítica...")
    analytic = build_analytic_score(bases=bases)

    # --- 3. INFERENCIA: puntúa con los artefactos ya entrenados, etiqueta con
    # los cortes ya congelados, separa cédulas nuevas y prioriza las Alto ---
    periodo = str(analytic["Periodo"].iloc[0])
    print(f"\n[3/3] Infiriendo scoring (periodo {periodo})...")
    ejecutar_inferencia(analytic=analytic, periodo=periodo)

    print("\nInferencia -- Finalizada ✅")

if __name__ == "__main__":
    main()
