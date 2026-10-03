from pathlib import Path
import pandas as pd

from src.config import load_config, get_path, ON_KAGGLE


def _resolve_raw_file(relative: str, cfg: dict) -> Path:
    """Trouve un fichier brut : d'abord dans data/raw (local), sinon dans /kaggle/input."""
    local = get_path("raw", cfg) / relative
    if local.exists():
        return local

    if ON_KAGGLE:
        matches = list(Path("/kaggle/input").rglob(Path(relative).name))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Fichier introuvable : {local}\n"
        "→ Télécharge OASIS depuis Kaggle et place le CSV dans data/raw/oasis/\n"
        "→ Vérifie que le nom correspond à config.yaml (section data)."
    )


def load_oasis_longitudinal(cfg: dict | None = None) -> pd.DataFrame:
    """Charge le fichier OASIS longitudinal brut, sans aucune modification."""
    cfg = cfg or load_config()
    path = _resolve_raw_file(cfg["data"]["oasis_longitudinal"], cfg)
    df = pd.read_csv(path)

    # contrôle minimal : les colonnes indispensables sont bien là
    requises = [cfg["features"]["id"], cfg["target"]["source_column"]]
    manquantes = [c for c in requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {manquantes}")

    return df