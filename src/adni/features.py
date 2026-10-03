"""Table multimodale à la baseline : une ligne par patient de la cohorte.

"""
import pandas as pd

MOIS = 30.44  # jours moyens par mois

# Variables IRM (versions harmonisées "combat" : effet scanner/site corrigé)
_IRM = {
    "hippocampe": ["Left_Hippocampus_combat", "Right_Hippocampus_combat"],
    "amygdale": ["Left_Amygdala_combat", "Right_Amygdala_combat"],
    "ventricules": ["Left_Lateral_Ventricle_combat", "Right_Lateral_Ventricle_combat"],
    "entorhinal_vol": ["lh_entorhinal_volume_combat", "rh_entorhinal_volume_combat"],
    "fusiforme_vol": ["lh_fusiform_volume_combat", "rh_fusiform_volume_combat"],
    "cerveau_total": ["BrainSegVolNotVent_combat"],
}
_IRM_EPAISSEUR = {"entorhinal_ep": ["lh_entorhinal_thickness_combat", "rh_entorhinal_thickness_combat"]}
_ICV = "EstimatedTotalIntraCranialVol_combat"


def _prepare(name: str, df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """Transformations propres à certaines modalités. Renvoie (table, colonnes à garder)."""
    if name == "irm":
        out = df.copy()
        cols = []
        for nom, src in _IRM.items():               # volumes rapportés au crâne (en ‰)
            out[f"irm_{nom}"] = out[src].sum(axis=1) / out[_ICV] * 1000
            cols.append(f"irm_{nom}")
        for nom, src in _IRM_EPAISSEUR.items():     # épaisseur corticale moyenne (mm)
            out[f"irm_{nom}"] = out[src].mean(axis=1)
            cols.append(f"irm_{nom}")
        return out, cols
    if name == "pet_fdg":
        out = df[df["ROINAME"] == "MetaROI"].rename(columns={"MEAN": "fdg_metaroi"})
        return out, ["fdg_metaroi"]
    return df, list(spec["features"])


def _closest_measure(df, date_col, cols, cohort, cfg, prefix):
    """Mesure la plus proche de la baseline, dans la fenêtre autorisée."""
    id_col = cfg["keys"]["id"]
    lo, hi = cfg["baseline"]["window_days"]
    m = df[[id_col, date_col] + cols].copy()
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
    m = m.merge(cohort[[id_col, "date_baseline", "mois_conversion"]], on=id_col)
    m["ecart_j"] = (m[date_col] - m["date_baseline"]).dt.days
    m = m[(m["ecart_j"] >= lo) & (m["ecart_j"] <= hi)]
    # anti-fuite : jamais de mesure prise à la date de conversion ou après
    apres_conversion = m["ecart_j"] >= m["mois_conversion"] * MOIS
    m = m[~apres_conversion].dropna(subset=cols, how="all")
    m = m.loc[m.groupby(id_col)["ecart_j"].apply(lambda s: s.abs().idxmin())]
    renommage = {c: c if c.startswith(("irm_", "fdg_")) else f"{prefix}_{c}" for c in cols}
    return m[[id_col] + cols].rename(columns=renommage)


def apoe4_count(apoe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Nombre d'allèles APOE e4 (0, 1 ou 2) : principal facteur de risque génétique."""
    id_col = cfg["keys"]["id"]
    g = apoe[[id_col, "GENOTYPE"]].dropna().drop_duplicates(id_col)
    g["clin_apoe4"] = g["GENOTYPE"].astype(str).str.count("4")
    return g[[id_col, "clin_apoe4"]]


def build_baseline_table(tables: dict, cohort: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    id_col = cfg["keys"]["id"]
    base = cohort.copy()
    base["date_baseline"] = pd.to_datetime(base["date_baseline"])
    out = base.rename(columns={"age": "clin_age", "sexe": "clin_sexe", "education": "clin_education"})
    out = out.merge(apoe4_count(tables["apoe"], cfg), on=id_col, how="left")

    for name, spec in cfg["baseline"]["modalities"].items():
        df, cols = _prepare(name, tables[spec["table"]], spec)
        mesure = _closest_measure(df, spec["date"], cols, base, cfg, prefix=name)
        out = out.merge(mesure, on=id_col, how="left")
    return out


def modality_columns(df: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Regroupe les colonnes de la table par modalité (utile pour RQ1 et RQ3)."""
    groupes = {"clinique": [c for c in df.columns if c.startswith("clin_")]}
    for name in cfg["baseline"]["modalities"]:
        if name == "irm":
            groupes[name] = [c for c in df.columns if c.startswith("irm_")]
        elif name == "pet_fdg":
            groupes[name] = ["fdg_metaroi"]
        else:
            groupes[name] = [c for c in df.columns if c.startswith(f"{name}_")]
    return groupes