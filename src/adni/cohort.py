"""Construction de la cohorte et de la cible : MCI -> Alzheimer à 36 mois.

Cible (une ligne par patient) :
  1 = pMCI : MCI au départ, diagnostiqué Alzheimer dans les 36 mois
  0 = sMCI : MCI au départ, toujours non dément après au moins 36 mois de suivi

Exclus (cible incertaine ou hors sujet) :
  - suivi trop court sans conversion (on ne sait pas ce qui se passe ensuite)
  - conversion après 36 mois (ni clairement pMCI, ni clairement sMCI)
  - démence non-Alzheimer à un moment du suivi
"""
import pandas as pd

from src.adni.load import load_adni_config, load_table

MOIS = 30.44  # jours moyens par mois


def _visites_valides(dx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cohort"]
    id_col, visit_col = cfg["keys"]["id"], cfg["keys"]["visit"]
    d = dx.copy()
    d = d[~d[visit_col].isin(c["excluded_visits"])]            # pas de screen failure
    d = d.dropna(subset=[c["diagnosis_column"]])                  # diagnostic connu
    d[c["date_column"]] = pd.to_datetime(d[c["date_column"]], errors="coerce")
    d = d.dropna(subset=[c["date_column"]])
    return d.sort_values([id_col, c["date_column"]]).reset_index(drop=True)


def build_mci_cohort(dx: pd.DataFrame, cfg: dict | None = None):
    """Renvoie (cohorte, bilan) : la table patient-cible et le décompte des exclusions."""
    cfg = cfg or load_adni_config()
    c = cfg["cohort"]
    id_col, visit_col = cfg["keys"]["id"], cfg["keys"]["visit"]
    dxc, datec = c["diagnosis_column"], c["date_column"]
    H, tol = c["horizon_months"], c["tolerance_months"]

    bilan = {"patients (table diagnostic)": dx[id_col].nunique()}
    d = _visites_valides(dx, cfg)
    bilan["avec au moins un diagnostic valide"] = d[id_col].nunique()

    # 1) baseline = première visite valide ; on garde les MCI
    base = d.groupby(id_col).first()
    mci_ids = base.index[base[dxc] == c["baseline_dx"]]
    d = d[d[id_col].isin(mci_ids)].copy()
    bilan["MCI au départ"] = len(mci_ids)

    # 2) exclure toute démence non-Alzheimer pendant le suivi
    non_ad = d.loc[d[dxc].isin(c["excluded_dx"]), id_col].unique()
    d = d[~d[id_col].isin(non_ad)]
    bilan["après exclusion démence non-Alzheimer"] = d[id_col].nunique()

    # 3) temps depuis la baseline
    d["mois"] = (d[datec] - d.groupby(id_col)[datec].transform("min")).dt.days / MOIS
    conv = d[d[dxc] == c["converted_dx"]].groupby(id_col)["mois"].min()
    suivi = d.groupby(id_col)["mois"].max()

    pat = pd.DataFrame({"mois_conversion": conv, "suivi_mois": suivi})
    pat["target"] = pd.NA
    pat.loc[pat["mois_conversion"] <= H + tol, "target"] = 1                       # pMCI
    pat.loc[pat["mois_conversion"].isna() & (pat["suivi_mois"] >= H - tol), "target"] = 0  # sMCI

    exclus_suivi = pat["mois_conversion"].isna() & (pat["suivi_mois"] < H - tol)
    exclus_tardif = pat["mois_conversion"] > H + tol
    bilan["exclus : suivi < 36 mois sans conversion"] = int(exclus_suivi.sum())
    bilan["exclus : conversion après 36 mois"] = int(exclus_tardif.sum())

    pat = pat.dropna(subset=["target"])
    pat["target"] = pat["target"].astype(int)

    # 4) informations de baseline utiles (phase ADNI pour la validation externe)
    infos = base.loc[pat.index, [visit_col, datec, c["phase_column"],
                                 "PHC_Age_Diagnosis", "PHC_Sex", "PHC_Education"]]
    infos = infos.rename(columns={visit_col: "visite_baseline", datec: "date_baseline",
                                  c["phase_column"]: "phase", "PHC_Age_Diagnosis": "age",
                                  "PHC_Sex": "sexe", "PHC_Education": "education"})
    cohorte = infos.join(pat).reset_index()

    bilan["COHORTE FINALE"] = len(cohorte)
    bilan["  dont pMCI (cible = 1)"] = int((cohorte["target"] == 1).sum())
    bilan["  dont sMCI (cible = 0)"] = int((cohorte["target"] == 0).sum())
    return cohorte, bilan