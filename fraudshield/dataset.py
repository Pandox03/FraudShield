"""Télécharge et prépare le jeu de données : python -m fraudshield.dataset"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from fraudshield.config import DATA_PATH, DATA_URL, TARGET


def download_data(force: bool = False) -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"Fichier déjà présent : {DATA_PATH}")
        return DATA_PATH
    print(f"Téléchargement depuis {DATA_URL} ...")
    urlretrieve(DATA_URL, DATA_PATH)
    print(f"Enregistré dans {DATA_PATH}")
    return DATA_PATH


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        download_data()
    df = pd.read_csv(path)
    return df


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    n_fraud = int(df[TARGET].sum())
    return {
        "lignes": n,
        "colonnes": list(df.columns),
        "fraudes": n_fraud,
        "legitimes": n - n_fraud,
        "taux_fraude": round(n_fraud / n, 5),
        "montant_moyen": round(float(df["Amount"].mean()), 2),
        "montant_max": round(float(df["Amount"].max()), 2),
    }


if __name__ == "__main__":
    download_data()
    df = load_data()
    info = summarize(df)
    print(f"{info['lignes']} transactions")
    print(f"Fraudes : {info['fraudes']} ({info['taux_fraude']:.3%})")
    print(f"Légitimes : {info['legitimes']}")
    print(f"Montant moyen : {info['montant_moyen']} | max : {info['montant_max']}")
