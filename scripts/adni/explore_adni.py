"""Exploration complète des données ADNI : figures et tableaux pour le rapport.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import get_path
from src.adni.load import load_adni_config, load_all
from src.adni import eda


def main() -> None:
    cfg = load_adni_config()
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed_baseline"])
    figs = get_path("results", cfg) / "figures"
    tabs = get_path("results", cfg) / "tables"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)

    # 1) résumé des variables
    resume = eda.resume_variables(df, cfg)
    resume.to_csv(tabs / "eda_resume_variables.csv", index=False)
    pd.set_option("display.width", 200)
    print("1) Résumé des variables (triées par pouvoir séparateur) :\n")
    print(resume.sort_values("auc_univariee", ascending=False)[
        ["variable", "manquant_%", "moy_stables", "moy_convertis", "cohen_d",
         "auc_univariee", "valeurs_extremes"]].to_string(index=False))

    # 2) figures
    eda.fig_composition(df, figs / "eda_1_composition_phases.png")
    eda.fig_couverture(df, cfg, figs / "eda_2_couverture_phases.png")
    eda.fig_distributions(df, cfg, figs / "eda_3_distributions.png")
    corr = eda.fig_correlations(df, cfg, figs / "eda_4_correlations.png")
    eda.fig_auc_univariee(resume, figs / "eda_5_auc_univariee.png")

    # 3) paires de variables très corrélées (redondance)
    masque = pd.DataFrame([[i <= j for j in range(len(corr))] for i in range(len(corr))],
                          index=corr.index, columns=corr.columns)
    paires = corr.where(~masque).stack()
    fortes = paires[paires.abs() >= 0.7].sort_values(key=abs, ascending=False)
    print("\n2) Paires de variables très corrélées (|r| >= 0,7) :")
    for (a, b), r in fortes.items():
        print(f"   {eda.ETIQUETTES.get(a, a):30s} ~ {eda.ETIQUETTES.get(b, b):30s} r = {r:+.2f}")

    # 4) exploration longitudinale (préparation de RQ2)
    longi = eda.exploration_longitudinale(load_all(cfg), df, cfg)
    longi.to_csv(tabs / "eda_longitudinal.csv", index=False)
    print("\n3) Mesures répétées avant conversion (préparation de RQ2) :\n")
    print(longi.to_string(index=False))

    print(f"\nFigures enregistrées dans : {figs}")
    print(f"Tableaux enregistrés dans : {tabs}")


if __name__ == "__main__":
    main()