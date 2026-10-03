"""Modèles et protocole d'évaluation pour ADNI.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.adni.features import modality_columns

MODELES = {
    "Régression logistique": lambda seed: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed)),
    "Forêt aléatoire": lambda seed: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               random_state=seed, n_jobs=-1)),
}


def grouped_modalities(df: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Colonnes par modalité, avec les regroupements définis dans la config."""
    groupes = modality_columns(df, cfg)
    for cible, sources in cfg["evaluation"]["merge_modalities"].items():
        groupes[cible] = sum((groupes.pop(s) for s in sources), [])
    return groupes


def cv_splits(y, cfg):
    e = cfg["evaluation"]
    cv = RepeatedStratifiedKFold(n_splits=e["cv_folds"], n_repeats=e["cv_repeats"],
                                 random_state=cfg["project"]["seed"])
    return list(cv.split(np.zeros(len(y)), y))


def cv_auc(make_model, X: pd.DataFrame, y: pd.Series, splits, seed: int) -> np.ndarray:
    """AUC sur chaque découpage (renvoie un tableau de 25 valeurs)."""
    scores = []
    for tr, te in splits:
        m = make_model(seed).fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(scores)