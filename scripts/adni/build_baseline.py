"""Construit la table multimodale à la baseline (une ligne par patient).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import get_path
from src.adni.load import load_adni_config, load_all
from src.adni.features import build_baseline_table, modality_columns


def main() -> None:
    cfg = load_adni_config()
    cohort = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["cohort"])
    table = build_baseline_table(load_all(cfg), cohort, cfg)

    groupes = modality_columns(table, cfg)
    print(f"Table multimodale : {table.shape[0]} patients x {table.shape[1]} colonnes\n")
    print("Disponibilité de chaque modalité à la baseline :")
    lignes = []
    for mod, cols in groupes.items():
        dispo = table[cols].notna().any(axis=1)
        lignes.append({"modalite": mod, "variables": len(cols),
                       "patients": int(dispo.sum()), "couverture": f"{dispo.mean():.0%}"})
    print(pd.DataFrame(lignes).to_string(index=False))

    out = get_path("processed", cfg) / cfg["processed_baseline"]
    table.to_parquet(out, index=False)
    print(f"\nTable enregistrée : {out}")


if __name__ == "__main__":
    main()