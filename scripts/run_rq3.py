"""RQ3 : robustesse aux modalites manquantes. Lancement : python scripts/run_rq3.py"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import load_config, get_path
from src.evaluation.robustness import robustness_to_missing_modality

def main():
    cfg = load_config()
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])
    res = robustness_to_missing_modality(df, cfg)
    complet = res.loc[res["scenario"].str.startswith("Complet"), "auc_moyenne"].iloc[0]
    res["chute_vs_complet"] = (res["auc_moyenne"] - complet).round(3)
    print("RQ3 - robustesse aux modalites manquantes (modele statique) :\n")
    print(res.to_string(index=False))
    out = get_path("results", cfg) / "tables" / "rq3_modalites_manquantes.csv"
    res.to_csv(out, index=False)
    print(f"\nTableau sauvegarde : {out}")

if __name__ == "__main__":
    main()