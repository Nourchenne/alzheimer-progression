"""Chargement des données brutes ADNI."""
from pathlib import Path
import pandas as pd

from src.config import ROOT, ON_KAGGLE, load_config, get_path

ADNI_CONFIG = ROOT / "config" / "adni.yaml"


def load_adni_config() -> dict:
    return load_config(ADNI_CONFIG)


def _resolve(relative: str, cfg: dict) -> Path:
    local = get_path("raw", cfg) / relative
    if local.exists():
        return local
    if ON_KAGGLE:
        matches = list(Path("/kaggle/input").rglob(Path(relative).name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Fichier introuvable : {local}\n"
        "-> Vérifie que le fichier est bien rangé dans data/raw/adni/ "
        "et que son nom correspond à config/adni.yaml (section tables)."
    )


def load_table(name: str, cfg: dict | None = None) -> pd.DataFrame:
    """Charge une table ADNI par son nom logique (ex. 'demodx', 'mri')."""
    cfg = cfg or load_adni_config()
    if name not in cfg["tables"]:
        raise KeyError(f"Table inconnue '{name}'. Tables disponibles : {list(cfg['tables'])}")
    df = pd.read_csv(_resolve(cfg["tables"][name], cfg), low_memory=False)

    id_col, visit_col = cfg["keys"]["id"], cfg["keys"]["visit"]
    if id_col not in df.columns:
        raise ValueError(f"Table '{name}' : colonne {id_col} absente")
    if name not in cfg.get("per_patient_tables", []) and visit_col not in df.columns:
        raise ValueError(f"Table '{name}' : colonne {visit_col} absente")
    return df


def load_all(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Charge toutes les tables déclarées dans la config."""
    cfg = cfg or load_adni_config()
    return {name: load_table(name, cfg) for name in cfg["tables"]}