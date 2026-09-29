"""
Méthode rigoureuse : validation croisée PAR PATIENT + imputation dans le pipeline
(donc apprise uniquement sur l'entraînement de chaque pli).
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.split import patient_cv


def build_model(seed: int):
    """Pipeline standard : imputation médiane -> standardisation -> régression logistique."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )


def modality_groups(cfg: dict) -> dict[str, list[str]]:
    """Les modalités à comparer + la version multimodale (toutes réunies)."""
    f = cfg["features"]
    groups = {
        "Imagerie (IRM)": f["imaging"],
        "Cognitif": f["cognitive"],
        "Clinique": f["clinical"],
    }
    groups["MULTIMODAL"] = f["imaging"] + f["cognitive"] + f["clinical"]
    return groups


def compare_modalities(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Entraîne une baseline par modalité et renvoie un tableau d'AUC (RQ1)."""
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    y = df[cfg["target"]["column"]]
    groups = df[cfg["features"]["id"]]
    cv = patient_cv(cfg)

    lignes = []
    for nom, cols in modality_groups(cfg).items():
        scores = cross_val_score(
            build_model(seed), df[cols], y,
            cv=cv, groups=groups, scoring="roc_auc",
        )
        lignes.append({
            "modalite": nom,
            "n_variables": len(cols),
            "auc_moyenne": round(scores.mean(), 3),
            "ecart_type": round(scores.std(), 3),
        })
    return pd.DataFrame(lignes)