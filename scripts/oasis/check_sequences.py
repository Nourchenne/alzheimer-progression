"""Vérifie la construction des séquences longitudinales.

"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import load_config, get_path
from src.oasis.features.sequences import build_sequences, sequence_summary


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])

    X_seq, y, ids = build_sequences(df, cfg)

    print("Résumé des séquences :")
    print(sequence_summary(X_seq, y).to_string(index=False))

    print("\nExemple — patient", ids[0], ":")
    print("  nombre de visites :", len(X_seq[0]))
    print("  forme (visites x variables) :", X_seq[0].shape)


if __name__ == "__main__":
    main()