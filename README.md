# Prédiction de la progression de la maladie d'Alzheimer
**Projet Big Data & Deep Learning — approche multimodale et longitudinale (ADNI, OASIS)**

## Objectif
Exploiter conjointement IRM, PET, données cognitives, cliniques et biomarqueurs pour prédire la progression vers Alzheimer.

## Questions de recherche
- **RQ1** — L'intégration multimodale améliore-t-elle la prédiction par rapport à l'unimodal ?
- **RQ2** — Plusieurs visites longitudinales améliorent-elles la prédiction par rapport à une visite unique ?
- **RQ3** — Le modèle reste-t-il robuste face aux données manquantes (modalité ou visite absente) ?

## Structure du projet
```
config/          Configuration centrale (chemins, variables, cible)
data/
  raw/           Données brutes, jamais modifiées (oasis/, oasis3/, adni/)
  interim/       Étapes intermédiaires
  processed/     Données propres, prêtes pour les modèles (contrat de données)
notebooks/       Explorations numérotées (01_ à 04_)
src/             Code source réutilisable
  data/          Chargement, nettoyage, découpage
  pipeline/      Pipeline Big Data (PySpark)
  features/      Séquences longitudinales
  models/        Baselines, multimodal, longitudinal
  evaluation/    Métriques, robustesse
  utils/         Fonctions partagées
scripts/         Points d'entrée (prétraitement, entraînement)
models/          Modèles entraînés sauvegardés (non versionnés)
results/         Figures, tableaux, logs
reports/         Rapport final et présentation
docs/            Documents de cadrage et d'organisation
tests/           Tests du code
```

## Répartition du binôme
| Membre A — Données & Pipeline | Membre B — Modélisation & Deep Learning |
|---|---|
| `src/data`, `src/pipeline`, `src/features` | `src/models`, `src/evaluation` |

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Données
Les données **ne sont pas dans le dépôt** (volume + conditions d'utilisation ADNI/OASIS).
Les placer manuellement dans `data/raw/<cohorte>/`.
- OASIS (Kaggle) : https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
- OASIS-3 : https://sites.wustl.edu/oasisbrains/request-access/
- ADNI : https://adni.loni.usc.edu/data-samples/adni-data/

## Règles de travail
- Une branche par tâche, jamais de commit direct sur `main`.
- Fusion dans `main` via Pull Request relue par l'autre membre.
- Aucun chemin en dur : tout passe par `config/config.yaml`.
