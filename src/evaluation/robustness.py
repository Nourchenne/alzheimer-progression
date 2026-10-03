"""[Membre B] RQ3 : robustesse aux donnees manquantes (modalite ou visite absente)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from src.config import load_config
from src.data.split import patient_cv
from src.models.baselines import build_model, first_visit_only
from src.features.sequences import build_sequences
from src.models.longitudinal import (
    set_seed, fit_stats, apply_stats, to_padded_batch, LSTMClassifier,
)

# --- Partie 1 : modalite manquante (modele statique) ---
def robustness_to_missing_modality(df, cfg=None):
    cfg = cfg or load_config()
    f = cfg["features"]
    base = first_visit_only(df, cfg)
    y = base[cfg["target"]["column"]]
    groups = base[cfg["features"]["id"]]
    all_feats = f["imaging"] + f["cognitive"] + f["clinical"]
    scenarios = {
        "Complet (aucune absente)": [],
        "sans Imagerie": f["imaging"],
        "sans Cognitif": f["cognitive"],
        "sans Clinique": f["clinical"],
    }
    lignes = []
    for nom, a_retirer in scenarios.items():
        X = base[all_feats].copy()
        X[a_retirer] = 0.0
        s = cross_val_score(build_model(cfg["project"]["seed"]), X, y,
                            cv=patient_cv(cfg), groups=groups, scoring="roc_auc")
        lignes.append({"scenario": nom,
                        "auc_moyenne": round(s.mean(), 3),
                        "ecart_type": round(s.std(), 3)})
    return pd.DataFrame(lignes)