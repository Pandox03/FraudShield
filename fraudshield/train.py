"""Entraîne et compare plusieurs modèles : python -m fraudshield.train"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from fraudshield.config import (
    ALERT_THRESHOLD,
    COMPARISON_PATH,
    EDA_PATH,
    FEATURES,
    FIGURES_DIR,
    INFO_PATH,
    MODEL_PATH,
    SAMPLES_PATH,
    SEED,
    TARGET,
)
from fraudshield.dataset import load_data, summarize


def _metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "rappel": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "vp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "vn": int(tn),
    }


def _build_candidates():
    """Trois approches différentes face au déséquilibre."""
    return {
        "logreg_sans_ponderation": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        ),
        "logreg_class_weight": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.08,
            max_iter=150,
            random_state=SEED,
            class_weight="balanced",
        ),
    }


def _undersample_train(X_tr, y_tr, ratio: float = 0.1):
    """Sous-échantillonage : garde toutes les fraudes + une fraction des légitimes."""
    df = X_tr.copy()
    df[TARGET] = y_tr.values
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    n_legit = max(len(fraud), int(len(fraud) / ratio))
    n_legit = min(n_legit, len(legit))
    legit_s = resample(legit, n_samples=n_legit, random_state=SEED, replace=False)
    bal = pd.concat([fraud, legit_s]).sample(frac=1.0, random_state=SEED)
    return bal[FEATURES], bal[TARGET]


def _save_eda_figures(df: pd.DataFrame) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # répartition des classes
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Légitime (0)", "Fraude (1)"], counts.values, color=["#4a5568", "#c53030"])
    ax.set_ylabel("Nombre de transactions")
    ax.set_title("Déséquilibre des classes")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    bal_path = FIGURES_DIR / "classes.png"
    fig.savefig(bal_path, dpi=120)
    plt.close(fig)

    # montants (échelle log)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(
        df.loc[df[TARGET] == 0, "Amount"].clip(upper=2000),
        bins=40,
        alpha=0.7,
        label="Légitime",
        color="#4a5568",
    )
    ax.hist(
        df.loc[df[TARGET] == 1, "Amount"].clip(upper=2000),
        bins=40,
        alpha=0.8,
        label="Fraude",
        color="#c53030",
    )
    ax.set_xlabel("Montant (tronqué à 2000)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des montants")
    ax.legend()
    fig.tight_layout()
    amt_path = FIGURES_DIR / "montants.png"
    fig.savefig(amt_path, dpi=120)
    plt.close(fig)

    eda = {
        "taux_fraude": round(float(df[TARGET].mean()), 5),
        "montant_moyen_legit": round(float(df.loc[df[TARGET] == 0, "Amount"].mean()), 2),
        "montant_moyen_fraude": round(float(df.loc[df[TARGET] == 1, "Amount"].mean()), 2),
        "montant_median_legit": round(float(df.loc[df[TARGET] == 0, "Amount"].median()), 2),
        "montant_median_fraude": round(float(df.loc[df[TARGET] == 1, "Amount"].median()), 2),
        "figures": {
            "classes": str(bal_path.name),
            "montants": str(amt_path.name),
        },
    }
    EDA_PATH.write_text(json.dumps(eda, indent=2, ensure_ascii=False), encoding="utf-8")
    return eda


def _save_comparison_chart(resultats: dict, meilleur: str):
    noms = list(resultats.keys())
    f1s = [resultats[n]["f1"] for n in noms]
    recalls = [resultats[n]["rappel"] for n in noms]
    pras = [resultats[n]["pr_auc"] for n in noms]

    x = np.arange(len(noms))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w, f1s, w, label="F1")
    ax.bar(x, recalls, w, label="Rappel")
    ax.bar(x + w, pras, w, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Comparaison des modèles (retenu : {meilleur})")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "comparaison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_confusion(y_true, y_pred, nom: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Légitime", "Fraude"], ax=ax, colorbar=False
    )
    ax.set_title(f"Matrice de confusion — {nom}")
    fig.tight_layout()
    path = FIGURES_DIR / "confusion.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train() -> dict:
    df = load_data()
    resume = summarize(df)
    eda = _save_eda_figures(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # variante undersampling + logreg (même jeu de test)
    X_us, y_us = _undersample_train(X_tr, y_tr)
    us_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )

    resultats = {}
    fitted = {}

    for nom, model in _build_candidates().items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= ALERT_THRESHOLD).astype(int)
        m = _metrics(y_te, pred, proba)
        resultats[nom] = m
        fitted[nom] = (model, proba, pred)
        print(f"\n=== {nom} ===")
        for k, v in m.items():
            print(f"  {k:>10} : {v}")

    us_model.fit(X_us, y_us)
    proba_us = us_model.predict_proba(X_te)[:, 1]
    pred_us = (proba_us >= ALERT_THRESHOLD).astype(int)
    resultats["logreg_undersampling"] = _metrics(y_te, pred_us, proba_us)
    fitted["logreg_undersampling"] = (us_model, proba_us, pred_us)
    print("\n=== logreg_undersampling ===")
    for k, v in resultats["logreg_undersampling"].items():
        print(f"  {k:>10} : {v}")

    # choix : meilleur F1, en cas d'égalité on préfère PR-AUC
    meilleur_nom = max(
        resultats.keys(),
        key=lambda n: (resultats[n]["f1"], resultats[n]["pr_auc"]),
    )
    meilleurs, proba_best, pred_best = fitted[meilleur_nom]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meilleurs, MODEL_PATH)

    cmp_df = pd.DataFrame(resultats).T
    cmp_df.index.name = "modele"
    cmp_df.to_csv(COMPARISON_PATH)

    _save_comparison_chart(resultats, meilleur_nom)
    _save_confusion(y_te, pred_best, meilleur_nom)

    echantillons = pd.concat(
        [
            df[df[TARGET] == 1].sample(n=min(15, int(df[TARGET].sum())), random_state=SEED),
            df[df[TARGET] == 0].sample(n=15, random_state=SEED),
        ]
    )
    echantillons.to_csv(SAMPLES_PATH, index=False)

    justification = (
        f"Le modèle « {meilleur_nom} » est retenu car il obtient le meilleur F1 "
        f"({resultats[meilleur_nom]['f1']}) sur le jeu de test stratifié, "
        f"avec un PR-AUC de {resultats[meilleur_nom]['pr_auc']}. "
        "L'accuracy n'est pas utilisée comme critère principal à cause du "
        f"déséquilibre ({resume['taux_fraude']:.3%} de fraudes)."
    )

    info = {
        "version": "1.1",
        "modele_retenu": meilleur_nom,
        "justification": justification,
        "features": FEATURES,
        "cible": TARGET,
        "seuil": ALERT_THRESHOLD,
        "dataset": resume,
        "eda": eda,
        "comparaison": resultats,
        "metriques": resultats[meilleur_nom],
        "matrice_confusion": {
            "vp": resultats[meilleur_nom]["vp"],
            "fp": resultats[meilleur_nom]["fp"],
            "fn": resultats[meilleur_nom]["fn"],
            "vn": resultats[meilleur_nom]["vn"],
        },
        "figures": [
            "classes.png",
            "montants.png",
            "comparaison.png",
            "confusion.png",
        ],
    }
    INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModèle retenu : {meilleur_nom}")
    print(justification)
    print(f"Sauvegardé dans {MODEL_PATH}")
    return info


if __name__ == "__main__":
    train()
