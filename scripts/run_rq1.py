"""RQ1 : le multimodal améliore-t-il la prédiction ? -> tableau d'AUC.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config, get_path
from src.models.baselines import compare_modalities


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])

    resultats = compare_modalities(df, cfg)
    resultats = resultats.sort_values("auc_moyenne", ascending=False).reset_index(drop=True)

    print("RQ1 — comparaison des modalités (AUC, validation croisée par patient) :\n")
    print(resultats.to_string(index=False))

    out = get_path("results", cfg) / "tables" / "rq1_modalites.csv"
    resultats.to_csv(out, index=False)
    print(f"\nTableau sauvegardé : {out}")

    multi = resultats.loc[resultats["modalite"] == "MULTIMODAL", "auc_moyenne"].iloc[0]
    best_uni = resultats[resultats["modalite"] != "MULTIMODAL"]["auc_moyenne"].max()
    verdict = "OUI" if multi > best_uni else "NON"
    print(f"\nRQ1 → multimodal ({multi}) vs meilleur unimodal ({best_uni}) : gain ? {verdict}")


if __name__ == "__main__":
    main()