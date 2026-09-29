import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # pour pouvoir importer src

from src.config import load_config, get_path
from src.data.load import load_oasis_longitudinal
from src.data.clean import clean_oasis


def main() -> None:
    cfg = load_config()

    raw = load_oasis_longitudinal(cfg)
    print(f"Données brutes     : {raw.shape}")

    clean = clean_oasis(raw, cfg)
    print(f"Données nettoyées  : {clean.shape}")

    out = get_path("processed", cfg) / cfg["processed"]["oasis_clean"]
    clean.to_parquet(out, index=False)
    print(f"Sauvegardé dans    : {out}")

    print("\nRépartition de la cible :")
    print(clean[cfg["target"]["column"]].value_counts().sort_index())

    print("\nValeurs manquantes restantes (imputées plus tard, dans le modèle) :")
    print(clean.isna().sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()