from pathlib import Path


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