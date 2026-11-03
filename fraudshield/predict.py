"""Charge le modèle et prédit pour une transaction."""
import json
from pathlib import Path

import joblib
import pandas as pd

from fraudshield.config import ALERT_THRESHOLD, FEATURES, INFO_PATH, MODEL_PATH


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, info_path: Path = INFO_PATH):
        self.model = joblib.load(model_path)
        info = {}
        if Path(info_path).exists():
            info = json.loads(Path(info_path).read_text(encoding="utf-8"))
        self.version = info.get("version", "inconnue")
        self.threshold = info.get("seuil", ALERT_THRESHOLD)
        self.modele_retenu = info.get("modele_retenu", "inconnu")

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features], columns=FEATURES)
        proba = float(self.model.predict_proba(df)[0, 1])
        return {
            "probabilite_fraude": round(proba, 4),
            "alerte": proba >= self.threshold,
            "seuil": self.threshold,
            "model_version": self.version,
            "modele": self.modele_retenu,
        }
