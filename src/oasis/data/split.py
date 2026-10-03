import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from src.config import load_config


def patient_labels(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Une étiquette par patient : 1 s'il est 'progresse' à au moins une visite."""
    return df.groupby(cfg["features"]["id"])[cfg["target"]["column"]].max()


def split_by_patient(df: pd.DataFrame, cfg: dict | None = None):
    """Découpe train/test par patient, en gardant la proportion 0/1 (stratification)."""
    cfg = cfg or load_config()
    id_col = cfg["features"]["id"]
    labels = patient_labels(df, cfg)

    train_ids, test_ids = train_test_split(
        labels.index.to_numpy(),
        test_size=cfg["split"]["test_size"],
        stratify=labels.to_numpy(),
        random_state=cfg["project"]["seed"],
    )
    train = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    test = df[df[id_col].isin(test_ids)].reset_index(drop=True)

    # Garde-fou : aucun patient des deux côtés
    commun = set(train[id_col]) & set(test[id_col])
    if commun:
        raise RuntimeError(f"Fuite : {len(commun)} patient(s) à la fois en train et en test")
    return train, test


def patient_cv(cfg: dict | None = None) -> StratifiedGroupKFold:
    """Validation croisée par patient (à utiliser avec groups=df[id])."""
    cfg = cfg or load_config()
    return StratifiedGroupKFold(
        n_splits=cfg["split"]["cv_folds"],
        shuffle=True,
        random_state=cfg["project"]["seed"],
    )