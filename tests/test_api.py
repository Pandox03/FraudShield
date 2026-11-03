"""Tests de l'API. Lancer depuis FraudShield : python -m pytest"""
import pandas as pd
from fastapi.testclient import TestClient

from fraudshield.api.main import app, state
from fraudshield.config import FEATURES, SAMPLES_PATH


def _exemple() -> dict:
    if SAMPLES_PATH.exists():
        row = pd.read_csv(SAMPLES_PATH).iloc[0]
        return {c: float(row[c]) for c in FEATURES}
    # valeurs neutres si pas encore d'échantillons
    body = {"Time": 100.0, "Amount": 50.0}
    for i in range(1, 29):
        body[f"V{i}"] = 0.0
    return body


VALIDE = _exemple()


def post(client, **changements):
    return client.post("/v1/predict", json={**VALIDE, **changements})


def test_health_repond_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_info_contient_modele_retenu():
    with TestClient(app) as client:
        r = client.get("/v1/info")
        assert r.status_code == 200
        body = r.json()
        assert "modele_retenu" in body
        assert "metriques" in body
        assert "comparaison" in body


def test_health_modele_charge():
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True


def test_predict_valide_renvoie_200():
    with TestClient(app) as client:
        assert post(client).status_code == 200


def test_predict_forme_de_la_reponse():
    with TestClient(app) as client:
        data = post(client).json()
        assert set(data) == {
            "probabilite_fraude",
            "alerte",
            "seuil",
            "model_version",
            "modele",
        }


def test_predict_probabilite_entre_0_et_1():
    with TestClient(app) as client:
        assert 0 <= post(client).json()["probabilite_fraude"] <= 1


def test_montant_negatif_refuse():
    with TestClient(app) as client:
        assert post(client, Amount=-10).status_code == 422


def test_time_negatif_refuse():
    with TestClient(app) as client:
        assert post(client, Time=-1).status_code == 422


def test_champ_manquant_refuse():
    with TestClient(app) as client:
        body = {k: v for k, v in VALIDE.items() if k != "Amount"}
        assert client.post("/v1/predict", json=body).status_code == 422


def test_champ_inconnu_refuse():
    with TestClient(app) as client:
        assert post(client, extra="x").status_code == 422


def test_modele_absent_renvoie_503():
    with TestClient(app) as client:
        state.pop("predictor", None)
        assert post(client).status_code == 503
