# FraudShield

Prototype de détection de transactions bancaires frauduleuses  
Projet 3 — Développement et déploiement d'applications intelligentes

Réalisé par : Imad Guidouh, Zakaria Bejgana, Zini Ayoub, Fathallah Zyad, Zakraia Tattah

## Objectif

Aider un analyste à repérer des transactions suspectes à partir d'un modèle de classification.  
Ce n'est **pas** un bloqueur automatique de paiements.

## Contenu du prototype

- Chargement / analyse du dataset Credit Card Fraud (ULB / Kaggle)
- Comparaison de plusieurs modèles face au déséquilibre de classes :
  - régression logistique (avec / sans `class_weight`)
  - régression logistique + undersampling
  - Random Forest
  - HistGradientBoosting
- Métriques adaptées : précision, rappel, F1, ROC-AUC, PR-AUC + matrice de confusion
- API FastAPI (`/health`, `/v1/info`, `/v1/predict`, `/v1/predict/batch`)
- Interface Streamlit (données, modèles, prédiction avec seuil réglable)
- Tests pytest
- Docker / docker-compose

## Installation

```bash
cd FraudShield
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

```bash
python -m fraudshield.dataset
python -m fraudshield.train
```

## API

```bash
python -m uvicorn fraudshield.api.main:app --port 8001 --reload
```

Docs : http://127.0.0.1:8001/docs

## Interface

```bash
streamlit run streamlit_app.py
```

http://127.0.0.1:8501

## Tests

```bash
python -m pytest
```

## Docker

```bash
docker compose up --build
```

## Structure

```
FraudShield/
├── data/                 # dataset, échantillons, résumé EDA
├── models/               # modèle, métriques, figures
├── fraudshield/
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── config.py
│   └── api/
├── streamlit_app.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
