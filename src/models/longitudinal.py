"""
Compare "toutes les visites" (ce module) à "1re visite" (baselines.evaluate_static).
Même cible, mêmes patients, même validation croisée par patient.
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.data.split import patient_cv
from src.features.sequences import build_sequences


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# --- Préparation : imputation + standardisation, apprises sur le TRAIN seulement ---
def fit_stats(train_seqs):
    allv = np.concatenate(train_seqs, axis=0)          # toutes les visites du train empilées
    med = np.nanmedian(allv, axis=0)
    allv = np.where(np.isnan(allv), med, allv)
    mean = allv.mean(axis=0)
    std = allv.std(axis=0) + 1e-8                       # +epsilon : évite la division par 0
    return med, mean, std


def apply_stats(seqs, stats):
    med, mean, std = stats
    return [((np.where(np.isnan(s), med, s) - mean) / std).astype("float32") for s in seqs]


def to_padded_batch(seqs):
    """Séquences de longueurs variables -> tenseur (batch, max_visites, variables) + longueurs."""
    tensors = [torch.tensor(s) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(tensors, batch_first=True), lengths


# --- Le modèle ---
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        # le LSTM ignore les visites de remplissage grâce à pack_padded_sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                # dernier état "mémoire"
        return self.fc(self.drop(h_n[-1])).squeeze(1)  # un score par patient


# --- Entraînement d'un pli ---
def _train_one_fold(train_seqs, y_tr, test_seqs, y_te, n_features, cfg):
    set_seed(cfg["project"]["seed"]); p = cfg["longitudinal"]
    stats = fit_stats(train_seqs)
    Xtr, ltr = to_padded_batch(apply_stats(train_seqs, stats))
    Xte, lte = to_padded_batch(apply_stats(test_seqs, stats))
    ytr = torch.tensor(y_tr, dtype=torch.float32)

    model = LSTMClassifier(n_features, p["hidden"], p["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(p["epochs"]):
        opt.zero_grad()
        loss_fn(model(Xtr, ltr), ytr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte, lte)).numpy()
    return roc_auc_score(y_te, proba)


# --- Évaluation en validation croisée par patient ---
def evaluate_longitudinal(df, cfg=None):
    cfg = cfg or load_config()
    X_seq, y, ids = build_sequences(df, cfg)
    n_features = X_seq[0].shape[1]
    X_seq = np.array(X_seq, dtype=object)
    aucs = []
    for tr, te in patient_cv(cfg).split(np.zeros(len(y)), y, groups=ids):
        aucs.append(_train_one_fold(list(X_seq[tr]), y[tr], list(X_seq[te]), y[te], n_features, cfg))
    return np.array(aucs)