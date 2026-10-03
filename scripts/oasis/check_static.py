import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import load_config, get_path
from src.oasis.models.baselines import evaluate_static, first_visit_only

cfg = load_config()
df = pd.read_parquet(get_path("processed", cfg) / cfg["processed"]["oasis_clean"])

base = first_visit_only(df, cfg)
print("Patients (1re visite) :", len(base),
      "| cible :", base[cfg["target"]["column"]].value_counts().to_dict())

auc = evaluate_static(df, cfg)
print(f"AUC statique (CV par patient) : {auc.mean():.3f} (+/- {auc.std():.3f})")