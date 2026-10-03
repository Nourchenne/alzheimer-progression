"""[Membre B] RQ3 : robustesse aux donnees manquantes (modalite ou visite absente)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from src.config import load_config
from src.oasis.data.split import patient_cv
from src.oasis.models.baselines import build_model, first_visit_only
from src.oasis.features.sequences import build_sequences
from src.oasis.models.longitudinal import (
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

# --- Partie 2 : visites manquantes (modele longitudinal LSTM) ---
def robustness_to_missing_visits(df, cfg=None, ks=(1, 2, 3, None)):
    """Entraine le LSTM sur sequences completes, evalue sur sequences
    tronquees a k visites (None = toutes les visites)."""
    cfg = cfg or load_config()
    p = cfg["longitudinal"]
    X_seq, y, ids = build_sequences(df, cfg)
    n_features = X_seq[0].shape[1]
    X_seq = np.array(X_seq, dtype=object)
    par_k = {k: [] for k in ks}
    for tr, te in patient_cv(cfg).split(np.zeros(len(y)), y, groups=ids):
        train_seqs = list(X_seq[tr])
        test_full = list(X_seq[te])
        y_te = y[te]
        set_seed(cfg["project"]["seed"])
        stats = fit_stats(train_seqs)
        Xtr, ltr = to_padded_batch(apply_stats(train_seqs, stats))
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        model = LSTMClassifier(n_features, p["hidden"], p["dropout"])
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"],
                               weight_decay=p["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(p["epochs"]):
            opt.zero_grad()
            loss_fn(model(Xtr, ltr), ytr).backward()
            opt.step()
        model.eval()
        for k in ks:
            te_seqs = test_full if k is None else [s[:k] for s in test_full]
            Xte, lte = to_padded_batch(apply_stats(te_seqs, stats))
            with torch.no_grad():
                proba = torch.sigmoid(model(Xte, lte)).numpy()
            par_k[k].append(roc_auc_score(y_te, proba))
    lignes = []
    for k in ks:
        label = "Toutes les visites" if k is None else f"{k} visite(s) max"
        arr = np.array(par_k[k])
        lignes.append({"scenario": label,
                        "auc_moyenne": round(arr.mean(), 3),
                        "ecart_type": round(arr.std(), 3)})
    return pd.DataFrame(lignes)