"""
Transforme le tableau (une ligne = une visite) en :
  - X_seq : liste de séquences, une par patient, chaque séquence = visites ordonnées
  - y     : une étiquette par patient (1 s'il progresse à une visite quelconque)
  - ids   : l'identifiant de chaque patient (même ordre que X_seq et y)
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.oasis.data.clean import feature_columns


def build_sequences(df: pd.DataFrame, cfg: dict | None = None):
    """Regroupe les visites par patient en séquences ordonnées dans le temps."""
    cfg = cfg or load_config()
    id_col = cfg["features"]["id"]
    visit_col = cfg["features"]["time"][0]      # 'Visit'
    target_col = cfg["target"]["column"]
    feats = feature_columns(cfg)                # imagerie + cognitif + clinique

    X_seq, y, ids = [], [], []
    for pid, groupe in df.groupby(id_col):
        groupe = groupe.sort_values(visit_col)  # visites dans l'ordre chronologique
        X_seq.append(groupe[feats].to_numpy(dtype="float32"))
        y.append(int(groupe[target_col].max()))  # progresse si au moins une visite = 1
        ids.append(pid)

    return X_seq, np.array(y), ids


def sequence_summary(X_seq, y) -> pd.DataFrame:
    """Petit résumé pour vérifier la construction."""
    longueurs = [len(s) for s in X_seq]
    return pd.DataFrame({
        "n_patients": [len(X_seq)],
        "n_variables": [X_seq[0].shape[1] if X_seq else 0],
        "visite_min": [min(longueurs)],
        "visite_max": [max(longueurs)],
        "visite_moyenne": [round(float(np.mean(longueurs)), 2)],
        "taux_progression": [round(float(y.mean()), 3)],
    })