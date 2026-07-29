from pathlib import Path
import yaml

def load_config() -> dict:
    
    """
    Carga el archivo de configuración YAML del proyecto como diccionario.

    Lee el archivo configs/config.yml ubicado en la raíz del proyecto
    y retorna su contenido deserializado como un diccionario de Python.

    Returns:
        dict: Contenido completo del archivo config.yml como diccionario.

    Raises:
        FileNotFoundError: Si el archivo configs/config.yml no existe.
    """

    
    config_path = Path(__file__).parents[2] / "configs" / "config.yml"
    
    if not config_path.exists():
        
        raise FileNotFoundError(
            f"No se encontró el archivo config.yml en {config_path}\n"
            f"Crea uno basandote en el archivo config.yml.example"
        )
        
    with open(config_path, "r", encoding="utf-8") as r:
        return yaml.safe_load(r)