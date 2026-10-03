"""Vérifie que toutes les tables ADNI se chargent, et donne un premier état des lieux.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.adni.load import load_adni_config, load_all


def main() -> None:
    cfg = load_adni_config()
    tables = load_all(cfg)
    id_col = cfg["keys"]["id"]

    lignes = []
    for name, df in tables.items():
        lignes.append({
            "table": name,
            "lignes": len(df),
            "colonnes": df.shape[1],
            "patients": df[id_col].nunique(),
            "lignes_par_patient": round(len(df) / df[id_col].nunique(), 1),
        })
    print("Tables ADNI chargées :\n")
    print(pd.DataFrame(lignes).to_string(index=False))

    dx = tables["demodx"]
    print("\nDiagnostic (PHC_Diagnosis) : 1 = normal, 2 = MCI, "
          "3 = démence Alzheimer, 4 = démence non-Alzheimer")
    print(dx["PHC_Diagnosis"].value_counts(dropna=False).sort_index().to_string())
    print(f"\nPatients au total : {dx[id_col].nunique()}")


if __name__ == "__main__":
    main()