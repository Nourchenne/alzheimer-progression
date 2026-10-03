"""Construit la cohorte MCI -> Alzheimer à 36 mois et l'enregistre.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import get_path
from src.adni.load import load_adni_config, load_table
from src.adni.cohort import build_mci_cohort


def main() -> None:
    cfg = load_adni_config()
    cohorte, bilan = build_mci_cohort(load_table("demodx", cfg), cfg)

    print("Construction de la cohorte (diagramme de flux) :\n")
    for etape, n in bilan.items():
        print(f"  {etape:45s} {n:6d}")

    print("\nRépartition par phase ADNI (pour la validation externe) :")
    print(cohorte.groupby("phase")["target"].agg(patients="size", pMCI="sum").to_string())

    out = get_path("processed", cfg) / cfg["processed"]["cohort"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cohorte.to_parquet(out, index=False)
    print(f"\nCohorte enregistrée : {out}")


if __name__ == "__main__":
    main()