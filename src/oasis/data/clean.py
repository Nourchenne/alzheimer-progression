import pandas as pd

from src.config import load_config


def feature_columns(cfg: dict) -> list[str]:
    """Variables d'entrée du modèle : imagerie + cognitif + clinique."""
    f = cfg["features"]
    return f["imaging"] + f["cognitive"] + f["clinical"]


def clean_oasis(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    f, t, c = cfg["features"], cfg["target"], cfg["cleaning"]
    data = df.copy()

    # 1) Cible binaire : 0 = stable, 1 = progresse / atteint
    data[t["column"]] = data[t["source_column"]].map(t["mapping"])
    if data[t["column"]].isna().any():
        inconnues = data.loc[data[t["column"]].isna(), t["source_column"]].unique()
        raise ValueError(f"Catégories de cible non prévues dans config.yaml : {inconnues}")
    data[t["column"]] = data[t["column"]].astype(int)

    # 2) Encodage du sexe (M=1, F=0)
    data[c["sex_column"]] = data[c["sex_column"]].map(c["sex_mapping"])

    # 3) On ne garde que les colonnes du contrat de données
    #    -> Hand, MRI ID, Group et CDR disparaissent ici
    colonnes = [f["id"]] + f["time"] + feature_columns(cfg) + [t["column"]]
    data = data[colonnes]

    # 4) Garde-fou anti-fuite : aucune colonne exclue ne doit survivre
    fuites = [col for col in f["excluded"] if col in data.columns]
    if fuites:
        raise RuntimeError(f"Fuite de données : colonnes interdites présentes {fuites}")

    # 5) Tri par patient puis par visite (indispensable pour le longitudinal)
    data = data.sort_values([f["id"], f["time"][0]]).reset_index(drop=True)
    return data