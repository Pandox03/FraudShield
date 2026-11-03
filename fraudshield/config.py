"""Réglages du projet FraudShield."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
SAMPLES_PATH = ROOT / "data" / "echantillons.csv"
EDA_PATH = ROOT / "data" / "eda_summary.json"
MODEL_PATH = ROOT / "models" / "model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
COMPARISON_PATH = ROOT / "models" / "comparaison.csv"
FIGURES_DIR = ROOT / "models" / "figures"

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

TARGET = "Class"  # 1 = fraude, 0 = légitime
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", "Amount"] + PCA_FEATURES

ALERT_THRESHOLD = 0.5
SEED = 42
