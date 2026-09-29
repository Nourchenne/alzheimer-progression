"""Démonstration : découpage par VISITE (faux) vs par PATIENT (correct).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config, get_path
from src.data.clean import feature_columns
from src.data.split import split_by_patient, patient_cv


def make_model(seed: int):
    # l'imputation est DANS le pipeline -> médiane apprise sur l'entraînement uniquement
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])

    id_col, t = cfg["features"]["id"], cfg["target"]["column"]
    X, y, groups = df[feature_columns(cfg)], df[t], df[id_col]

    # 1) Vérification du découpage train/test par patient
    train, test = split_by_patient(df, cfg)
    print(f"Train : {train[id_col].nunique()} patients / {len(train)} visites")
    print(f"Test  : {test[id_col].nunique()} patients / {len(test)} visites")
    print(f"Patients en commun : {len(set(train[id_col]) & set(test[id_col]))}\n")

    # 2) Comparaison des deux façons de faire la validation croisée
    cv_visite = StratifiedKFold(n_splits=cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    auc_visite = cross_val_score(make_model(seed), X, y, cv=cv_visite, scoring="roc_auc")
    auc_patient = cross_val_score(make_model(seed), X, y, cv=patient_cv(cfg), groups=groups, scoring="roc_auc")

    print("AUC en validation croisée (5 plis) :")
    print(f"  Découpage par VISITE  (faux)    : {auc_visite.mean():.3f} (±{auc_visite.std():.3f})")
    print(f"  Découpage par PATIENT (correct) : {auc_patient.mean():.3f} (±{auc_patient.std():.3f})")


if __name__ == "__main__":
    main()