"""RQ1 sur ADNI : la fusion multimodale améliore-t-elle la prédiction MCI -> Alzheimer ?

Analyse principale sur les patients disposant de TOUTES les modalités :
tous les modèles sont comparés sur exactement les mêmes patients.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import get_path
from src.adni.load import load_adni_config
from src.adni.models import MODELES, grouped_modalities, cv_splits, cv_auc


def main() -> None:
    cfg = load_adni_config()
    seed = cfg["project"]["seed"]
    df = pd.read_parquet(get_path("processed", cfg) / cfg["processed_baseline"])

    groupes = grouped_modalities(df, cfg)
    complet = pd.concat({m: df[c].notna().any(axis=1) for m, c in groupes.items()}, axis=1).all(axis=1)
    data = df[complet].reset_index(drop=True)
    y = data["target"]
    print(f"Patients avec toutes les modalités : {len(data)} (dont {int(y.sum())} convertis)\n")

    groupes["MULTIMODAL"] = sum(groupes.values(), [])
    splits = cv_splits(y, cfg)

    lignes, scores = [], {}
    for nom_modele, make_model in MODELES.items():
        for mod, cols in groupes.items():
            s = cv_auc(make_model, data[cols], y, splits, seed)
            scores[(nom_modele, mod)] = s
            lignes.append({"modele": nom_modele, "modalite": mod, "n_variables": len(cols),
                           "auc_moyenne": round(s.mean(), 3), "ecart_type": round(s.std(), 3)})
    res = pd.DataFrame(lignes)

    for nom_modele in MODELES:
        r = res[res["modele"] == nom_modele].sort_values("auc_moyenne", ascending=False)
        print(f"=== {nom_modele} ===")
        print(r.drop(columns="modele").to_string(index=False))
        uni = r[r["modalite"] != "MULTIMODAL"].iloc[0]["modalite"]
        diff = scores[(nom_modele, "MULTIMODAL")] - scores[(nom_modele, uni)]
        print(f"-> multimodal vs meilleure modalité seule ({uni}) : gain moyen {diff.mean():+.3f}, "
              f"multimodal meilleur sur {(diff > 0).sum()}/{len(diff)} découpages\n")

    out = get_path("results", cfg) / "tables" / "rq1_adni.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Tableau sauvegardé : {out}")


if __name__ == "__main__":
    main()