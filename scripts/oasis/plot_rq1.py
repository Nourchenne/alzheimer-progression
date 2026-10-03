"""Graphique de RQ1 : AUC par modalité, avec barre d'erreur.

"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")            # backend sans fenêtre : enregistre directement l'image
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, get_path


def main() -> None:
    cfg = load_config()
    tables = get_path("results", cfg) / "tables"
    res = pd.read_csv(tables / "rq1_modalites.csv").sort_values("auc_moyenne")

    # couleur différente pour le multimodal, pour le mettre en valeur
    couleurs = ["#C55A11" if m == "MULTIMODAL" else "#2E5496" for m in res["modalite"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(res["modalite"], res["auc_moyenne"],
            xerr=res["ecart_type"], color=couleurs, capsize=4)

    # écrire la valeur au bout de chaque barre
    for y, (v, e) in enumerate(zip(res["auc_moyenne"], res["ecart_type"])):
        ax.text(v + e + 0.015, y, f"{v:.3f}", va="center", fontsize=10)

    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)  # 0.5 = hasard
    ax.set_xlabel("AUC (validation croisée par patient)")
    ax.set_xlim(0, 1)
    ax.set_title("RQ1 — Apport de la fusion multimodale (OASIS)")
    plt.tight_layout()

    out = get_path("results", cfg) / "figures" / "rq1_modalites.png"
    fig.savefig(out, dpi=150)
    print(f"Figure sauvegardée : {out}")


if __name__ == "__main__":
    main()