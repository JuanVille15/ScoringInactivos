import json
from pathlib import Path


def leer_cortes_scoring(cortes_path: str | None = None) -> dict:
    """Lee los cortes Bajo/Medio/Alto persistidos por `categorizar_score`.

    Solo `build_score.py` (entrenamiento, vía `categorizar_score`) calcula y
    sobreescribe estos cortes. Todo lo demás los lee tal cual, congelados:
    `caracterizar_score.py` (diagnóstico post-hoc de una corrida de
    entrenamiento) y `inference.py` (etiqueta Bajo/Medio/Alto en inferencia
    sin recalcular nada). Vive acá porque ambos módulos, que no se llaman
    entre sí, necesitan la misma lectura.

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


def periodo_mas_cercano(path_raiz: Path, archivo_requerido: str | None = None) -> str:
    """Encuentra la carpeta de periodo (YYYYMM) más reciente bajo `path_raiz`.

    Cada etapa del pipeline (raw, analytic) persiste sus corridas en una
    carpeta por periodo de ejecución (ej. `data/analytic/202608/`), así que ya
    no hay un único archivo fijo que leer: hay que resolver cuál periodo usar
    cuando no se pide uno explícito. El orden lexicográfico de nombres YYYYMM
    coincide con el orden cronológico, así que basta ordenar y tomar el último.

    Args:
        path_raiz: Carpeta que contiene una subcarpeta por periodo (ej.
            `data/analytic/` o `data/raw/`).
        archivo_requerido: Si se indica, solo se consideran las subcarpetas que
            contienen ese archivo (ej. 'analytic_score_base.parquet') — evita
            devolver una carpeta de periodo a medio escribir o de otra etapa.
            Si es None, cualquier subcarpeta cuenta.

    Returns:
        El nombre de la carpeta de periodo más reciente.

    Raises:
        FileNotFoundError: Si `path_raiz` no existe o no tiene ninguna
            subcarpeta de periodo que cumpla la condición.
    """
    if not path_raiz.exists():
        candidatos = []
    elif archivo_requerido:
        candidatos = sorted(
            p.name for p in path_raiz.iterdir()
            if p.is_dir() and (p / archivo_requerido).exists()
        )
    else:
        candidatos = sorted(p.name for p in path_raiz.iterdir() if p.is_dir())

    if not candidatos:
        raise FileNotFoundError(
            f"No hay ninguna carpeta de periodo en {path_raiz}. "
            "Corre primero la etapa correspondiente del pipeline."
        )

    return candidatos[-1]