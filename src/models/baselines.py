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

from src.data.clean import feature_columns


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


# ---------------------------------------------------------------------------
# RQ2 — référence "statique" : une seule visite (la première) par patient
# ---------------------------------------------------------------------------
def first_visit_only(df, cfg):
    """Garde uniquement la 1re visite de chaque patient (analyse statique)."""
    id_col = cfg["features"]["id"]
    visit_col = cfg["features"]["time"][0]
    return (df.sort_values(visit_col)
              .groupby(id_col, as_index=False)
              .head(1)
              .reset_index(drop=True))


def evaluate_static(df, cfg=None):
    """AUC du modele statique (1re visite, multimodal) en CV par patient."""
    cfg = cfg or load_config()
    base = first_visit_only(df, cfg)
    X = base[feature_columns(cfg)]
    y = base[cfg["target"]["column"]]
    groups = base[cfg["features"]["id"]]
    return cross_val_score(
        build_model(cfg["project"]["seed"]),
        X, y, cv=patient_cv(cfg), groups=groups, scoring="roc_auc",
    )