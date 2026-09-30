"""RQ2 : le longitudinal (LSTM) fait-il mieux que le statique (1re visite) ?
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import load_config, get_path
from src.models.baselines import evaluate_static
from src.models.longitudinal import evaluate_longitudinal


def main():
    cfg = load_config()
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])

    print("Calcul du modele STATIQUE (1re visite)...")
    stat = evaluate_static(df, cfg)
    print("Calcul du modele LONGITUDINAL (LSTM)...")
    longi = evaluate_longitudinal(df, cfg)

    res = pd.DataFrame({
        "modele": ["Statique (1re visite)", "Longitudinal (LSTM)"],
        "auc_moyenne": [round(stat.mean(), 3), round(longi.mean(), 3)],
        "ecart_type": [round(stat.std(), 3), round(longi.std(), 3)],
    })
    print("\nRQ2 — statique vs longitudinal (AUC, CV par patient) :\n")
    print(res.to_string(index=False))

    # --- Verdict qui tient compte de l'incertitude ---
    # Le gain n'est retenu que s'il dépasse le bruit (variabilite entre plis).
    gain = longi.mean() - stat.mean()
    seuil = max(stat.std(), longi.std())
    if gain > seuil:
        verdict = "OUI, gain significatif"
    elif gain > 0:
        verdict = "Match nul (gain dans le bruit)"
    else:
        verdict = "NON"
    print(f"\nGain du longitudinal : {gain:+.3f} (bruit +/-{seuil:.3f})  ->  RQ2 : {verdict}")

    out = get_path("results", cfg) / "tables" / "rq2_statique_vs_longitudinal.csv"
    res.to_csv(out, index=False)
    print(f"Tableau sauvegarde : {out}")


if __name__ == "__main__":
    main()