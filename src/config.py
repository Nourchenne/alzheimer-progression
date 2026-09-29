"""Chargement de la configuration + gestion des chemins (local ou Kaggle)."""
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]          # racine du projet
ON_KAGGLE = os.path.exists("/kaggle/input")         # True si on tourne sur Kaggle


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str, cfg: dict | None = None) -> Path:
    """Renvoie le chemin absolu d'un dossier déclaré dans config.yaml (section paths)."""
    cfg = cfg or load_config()
    return ROOT / cfg["paths"][key]
