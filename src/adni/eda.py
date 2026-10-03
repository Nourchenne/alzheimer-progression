"""Exploration des données ADNI (EDA) : tableaux et figures pour le rapport.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.adni.features import modality_columns, MOIS

ETIQUETTES = {
    "clin_age": "Âge", "clin_sexe": "Sexe", "clin_education": "Éducation (années)",
    "clin_apoe4": "Allèles APOE4", "cognitif_PHC_MEM": "Mémoire (composite)",
    "mmse_MMSCORE": "MMSE",
    "irm_hippocampe": "Hippocampe (‰ crâne)", "irm_amygdale": "Amygdale (‰ crâne)",
    "irm_ventricules": "Ventricules (‰ crâne)", "irm_entorhinal_vol": "Entorhinal vol. (‰)",
    "irm_fusiforme_vol": "Fusiforme vol. (‰)", "irm_cerveau_total": "Cerveau total (‰)",
    "irm_entorhinal_ep": "Épaisseur entorhinale (mm)",
    "pet_amyloide_PHC_CENTILOIDS": "Amyloïde PET (Centiloïdes)", "fdg_metaroi": "FDG PET (MetaROI)",
    "lcr_PHC_AB42": "LCR AB42", "lcr_PHC_Tau": "LCR Tau", "lcr_PHC_pTau181": "LCR pTau181",
}
NOMS_MODALITES = {"clinique": "Clinique", "cognitif": "Cognitif (mémoire)", "mmse": "MMSE",
                  "irm": "IRM", "pet_amyloide": "PET amyloïde", "pet_fdg": "PET FDG", "lcr": "LCR"}
BLEU, ORANGE = "#2E5496", "#C55A11"


def variables(df, cfg):
    return sum(modality_columns(df, cfg).values(), [])


def resume_variables(df, cfg) -> pd.DataFrame:
    """Une ligne par variable : valeurs manquantes, distribution, valeurs extrêmes, séparation."""
    y = df["target"]
    lignes = []
    for mod, cols in modality_columns(df, cfg).items():
        for c in cols:
            x = df[c]
            ok = x.notna()
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            extremes = ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum()
            auc = roc_auc_score(y[ok], x[ok]) if ok.sum() > 10 and x[ok].nunique() > 1 else np.nan
            s0, s1 = x[y == 0], x[y == 1]
            d = (s1.mean() - s0.mean()) / np.sqrt((s0.var() + s1.var()) / 2)
            lignes.append({
                "modalite": mod, "variable": ETIQUETTES.get(c, c), "colonne": c,
                "manquant_%": round(100 * (1 - ok.mean()), 1),
                "moyenne": round(x.mean(), 2), "ecart_type": round(x.std(), 2),
                "min": round(x.min(), 2), "max": round(x.max(), 2),
                "valeurs_extremes": int(extremes),
                "moy_stables": round(s0.mean(), 2), "moy_convertis": round(s1.mean(), 2),
                "cohen_d": round(d, 2),
                "auc_univariee": round(max(auc, 1 - auc), 3) if pd.notna(auc) else np.nan,
            })
    return pd.DataFrame(lignes)


# ------------------------------------------------------------------ figures
def fig_composition(df, out):
    t = df.groupby("phase")["target"].agg(stables=lambda s: (s == 0).sum(), convertis="sum")
    t = t.loc[["ADNI1", "ADNIGO", "ADNI2", "ADNI3"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index, t["stables"], color=BLEU, label="Stables (sMCI)")
    ax.bar(t.index, t["convertis"], bottom=t["stables"], color=ORANGE, label="Convertis (pMCI)")
    for i, (s, c) in enumerate(zip(t["stables"], t["convertis"])):
        ax.text(i, s + c + 4, f"{c / (s + c):.0%} convertis", ha="center", fontsize=9)
    ax.set_ylabel("Patients"); ax.set_title("Composition de la cohorte par phase ADNI")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_couverture(df, cfg, out):
    groupes = modality_columns(df, cfg)
    phases = ["ADNI1", "ADNIGO", "ADNI2", "ADNI3"]
    m = pd.DataFrame({mod: [df.loc[df["phase"] == p, cols].notna().any(axis=1).mean() for p in phases]
                      for mod, cols in groupes.items()}, index=phases).T
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(phases)), phases)
    ax.set_yticks(range(len(m)), [NOMS_MODALITES.get(i, i) for i in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.6 else "black")
    ax.set_title("Disponibilité des modalités selon la phase ADNI", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.05, label="Part des patients")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_distributions(df, cfg, out):
    cols = [c for c in variables(df, cfg) if c != "clin_sexe"]   # binaire : une boîte n'a pas de sens
    n = len(cols); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.8 * nrow))
    for ax, c in zip(axes.flat, cols):
        data = [df.loc[df["target"] == k, c].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 2})
        for patch, col in zip(bp["boxes"], (BLEU, ORANGE)):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticks([1, 2], ["Stables", "Convertis"], fontsize=8)
        ax.set_title(ETIQUETTES.get(c, c), fontsize=9)
        ax.tick_params(axis="y", labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Distribution des variables à la baseline : stables vs convertis", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_correlations(df, cfg, out):
    cols = [c for c in variables(df, cfg) if c != "clin_sexe"]
    corr = df[cols].corr(method="spearman")
    labels = [ETIQUETTES.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Corrélation de Spearman")
    ax.set_title("Corrélations entre variables (toutes modalités)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return corr


def fig_auc_univariee(resume, out):
    r = resume.dropna(subset=["auc_univariee"]).sort_values("auc_univariee")
    couleurs = {"clinique": "#7F7F7F", "cognitif": BLEU, "mmse": BLEU, "irm": "#548235",
                "pet_amyloide": ORANGE, "pet_fdg": "#BF8F00", "lcr": "#7030A0"}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(r["variable"], r["auc_univariee"], color=[couleurs.get(m, "grey") for m in r["modalite"]])
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    for y, v in enumerate(r["auc_univariee"]):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=8)
    presentes = [m for m in couleurs if m in set(r["modalite"])]
    ax.legend([Patch(color=couleurs[m]) for m in presentes],
              [NOMS_MODALITES.get(m, m) for m in presentes], loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0.45, 1); ax.set_xlabel("AUC de la variable seule (0,5 = hasard)")
    ax.set_title("Pouvoir séparateur de chaque variable, prise isolément")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ longitudinal
def exploration_longitudinale(tables, cohort, cfg, fenetres=(12, 24)) -> pd.DataFrame:
    """Nombre de mesures par patient dans les premiers mois de suivi, AVANT conversion."""
    id_col = cfg["keys"]["id"]
    base = cohort[[id_col, "date_baseline", "mois_conversion", "target"]].copy()
    base["date_baseline"] = pd.to_datetime(base["date_baseline"])
    lignes = []
    for mod, spec in cfg["baseline"]["modalities"].items():
        df = tables[spec["table"]]
        if mod == "pet_fdg":
            df = df[df["ROINAME"] == "MetaROI"]
        m = df[[id_col, spec["date"]]].copy()
        m["date"] = pd.to_datetime(m[spec["date"]], errors="coerce")
        m = m.merge(base, on=id_col)
        m["mois"] = (m["date"] - m["date_baseline"]).dt.days / MOIS
        m = m[(m["mois"] >= -6) & ~(m["mois"] >= m["mois_conversion"])]
        for F in fenetres:
            # patients encore non convertis à F mois (les seuls utilisables en longitudinal)
            eligibles = base[~(base["mois_conversion"] <= F)]
            n = m[m["mois"] <= F].drop_duplicates([id_col, "date"]).groupby(id_col).size()
            n = n.reindex(eligibles[id_col], fill_value=0)
            lignes.append({"modalite": mod, "fenetre_mois": F,
                           "patients_eligibles": len(eligibles),
                           "convertis_eligibles": int(eligibles["target"].sum()),
                           "mesures_mediane": float(n.median()),
                           "patients_>=2_mesures_%": round(100 * (n >= 2).mean(), 1)})
    return pd.DataFrame(lignes)