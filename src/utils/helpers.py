import json
from pathlib import Path

from src.utils.config import load_config


# ─── Versión del score ────────────────────────────────────────────────────────
# Un "score" = escaladores (.pkl) + cortes Bajo/Medio/Alto entrenados juntos
# sobre el mismo lote. Van siempre juntos: cortes de una versión con
# escaladores de otra darían categorías inconsistentes. La versión vigente
# vive en config.yml -> scoring.version; train.py escribe en esa versión e
# inference.py lee de esa versión.

def version_scoring() -> str:
    """Versión vigente del score (config.yml -> scoring.version), ej. 'v2'."""
    version = load_config()["scoring"].get("version")
    if not version:
        raise KeyError("Falta scoring.version en config.yml (ej. 'v2').")
    return str(version)


def path_artefactos(version: str | None = None) -> Path:
    """Carpeta de escaladores/encoders de una versión: ``models/score/{version}/``."""
    return Path.cwd() / "models" / "score" / (version or version_scoring())


def path_cortes(version: str | None = None) -> Path:
    """json de cortes de una versión: ``configs/scoring/cortes_scoring_{version}.json``."""
    return Path.cwd() / "configs" / "scoring" / f"cortes_scoring_{version or version_scoring()}.json"


def path_entrenamiento(version: str | None = None) -> Path:
    """Carpeta con todo lo que produce un entrenamiento (raw, analytic,
    scoring, priorización): ``data/train/{version}/``. Separada de
    data/raw, data/analytic y data/scoring para que reentrenar nunca pise lo
    que ya se entregó en inferencia."""
    return Path.cwd() / "data" / "train" / (version or version_scoring())


def leer_cortes_scoring(cortes_path: str | None = None, version: str | None = None) -> dict:
    """Lee los cortes Bajo/Medio/Alto persistidos por `categorizar_score`.

    Solo `build_score.py` (entrenamiento, vía `categorizar_score`) calcula y
    escribe estos cortes. Todo lo demás los lee tal cual, congelados:
    `caracterizar_score.py` (diagnóstico post-hoc de una corrida de
    entrenamiento) y `inference.py` (etiqueta Bajo/Medio/Alto en inferencia
    sin recalcular nada). Vive acá porque ambos módulos, que no se llaman
    entre sí, necesitan la misma lectura.

    Args:
        cortes_path: Ruta personalizada. Si es None, usa `path_cortes(version)`.
        version: Versión del score. Si es None, la vigente en config.yml.

    Returns:
        dict con claves 'bajo', 'medio', 'alto', cada una [limite_inf, limite_sup].

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(cortes_path) if cortes_path else path_cortes(version)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero train.py para esa versión.")
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