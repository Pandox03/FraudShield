"""FraudShield — console analyste fraude (UI simple, lisible)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

from fraudshield.config import (
    COMPARISON_PATH,
    FEATURES,
    FIGURES_DIR,
    INFO_PATH,
    MODEL_PATH,
    PCA_FEATURES,
    SAMPLES_PATH,
)
from fraudshield.predict import Predictor

API_URL = os.getenv("API_URL", "http://localhost:8001/v1/predict")
HEALTH_URL = API_URL.replace("/v1/predict", "/health")
PAGES = ["Tableau de bord", "File d'attente", "Analyse manuelle", "Performance"]

st.set_page_config(page_title="FraudShield", layout="wide", initial_sidebar_state="expanded")

# CSS minimal : on ne touche PAS aux widgets Streamlit (radio, segmented, dataframe)
st.markdown(
    """
<style>
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton,
[data-testid="stAppDeployButton"] { display: none !important; visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; }
.verdict {
  padding: 1rem 1.1rem; border-radius: 10px; border: 1px solid #cbd5e1;
  background: #f8fafc; margin: 0.6rem 0;
}
.verdict.high { background: #fef2f2; border-color: #fca5a5; }
.verdict.mid { background: #fff7ed; border-color: #fdba74; }
.verdict.low { background: #ecfdf5; border-color: #6ee7b7; }
.verdict .score { font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def load_info() -> dict[str, Any]:
    if not INFO_PATH.exists():
        st.error("Aucun modèle. Lancez : python -m fraudshield.train")
        st.stop()
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor(MODEL_PATH)


def risk_level(score: float, seuil: float) -> str:
    if score >= seuil:
        return "Élevé"
    if score >= max(0.2, seuil * 0.45):
        return "Modéré"
    return "Faible"


@st.cache_data
def load_queue() -> pd.DataFrame:
    raw = pd.read_csv(SAMPLES_PATH)
    predictor = get_predictor()
    rows = []
    for i, row in raw.iterrows():
        feats = {c: float(row[c]) for c in FEATURES}
        out = predictor.predict(feats)
        score = float(out["probabilite_fraude"])
        t = int(row["Time"])
        rows.append(
            {
                "id": f"TX-{10000 + int(i)}",
                "montant": round(float(row["Amount"]), 2),
                "temps_s": t,
                "heure": f"{(t // 3600) % 24:02d}:{(t // 60) % 60:02d}",
                "score": score,
                "label_reel": "Fraude" if int(row["Class"]) == 1 else "Légitime",
                "Class": int(row["Class"]),
                **{c: float(row[c]) for c in FEATURES},
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def api_ok() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200 and bool(r.json().get("model_loaded"))
    except requests.RequestException:
        return False


def call_api(payload: dict) -> dict | None:
    try:
        r = requests.post(API_URL, json=payload, timeout=8)
    except requests.RequestException:
        st.error("API injoignable (port 8001).")
        return None
    if r.status_code != 200:
        st.error(f"Erreur API {r.status_code}")
        return None
    return r.json()


def score_chart(df: pd.DataFrame, seuil: float):
    data = df.copy()
    data["niveau"] = data["score"].map(lambda s: risk_level(float(s), seuil))
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("id:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "niveau:N",
                scale=alt.Scale(
                    domain=["Faible", "Modéré", "Élevé"],
                    range=["#059669", "#ea580c", "#dc2626"],
                ),
                legend=alt.Legend(title="Niveau"),
            ),
            tooltip=["id", alt.Tooltip("score:Q", format=".1%"), "montant", "niveau"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [seuil]}))
        .mark_rule(color="#111827", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return (bars + rule).properties(height=280)


def show_verdict(proba: float, seuil: float, label_reel: str | None = None) -> None:
    niveau = risk_level(proba, seuil)
    css = {"Élevé": "high", "Modéré": "mid", "Faible": "low"}[niveau]
    st.markdown(
        f"""
        <div class="verdict {css}">
          <div><b>Niveau :</b> {niveau}</div>
          <div class="score">{proba:.1%}</div>
          <div>Seuil d'alerte : {seuil:.0%} (aide à la décision)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if label_reel is not None:
        pred = "Fraude" if proba >= seuil else "Légitime"
        st.caption(f"{'OK' if pred == label_reel else 'Écart'} vs label réel ({label_reel}).")


info = load_info()
seuil_defaut = float(info.get("seuil", 0.5))
online = api_ok()
queue = load_queue()

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "selected_tx" not in st.session_state:
    st.session_state.selected_tx = str(queue.iloc[0]["id"]) if len(queue) else None
if "reviewed" not in st.session_state:
    st.session_state.reviewed = set()
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

with st.sidebar:
    st.title("FraudShield")
    st.caption("Console analyste fraude")
    if online:
        st.success("API connectée", icon="✅")
    else:
        st.error("API hors ligne", icon="⚠️")

    page = st.radio("Navigation", PAGES, key="page")
    st.divider()
    seuil = st.slider("Seuil d'alerte", 0.05, 0.95, seuil_defaut, 0.05)
    st.caption("Au-dessus du seuil = revue prioritaire")
    st.divider()
    st.write(f"**Modèle :** {info.get('modele_retenu')}")
    st.write(f"**Version :** {info.get('version')}")
    st.caption(datetime.now().strftime("%d/%m/%Y %H:%M"))

n_alerte = int((queue["score"] >= seuil).sum())

# ------------------------ pages ------------------------
if page == "Tableau de bord":
    st.header("Tableau de bord")
    st.write("Priorisation des dossiers et état du modèle.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", f"{info['dataset'].get('lignes', 0):,}")
    c2.metric("Fraudes", f"{info['dataset'].get('fraudes', 0):,}")
    c3.metric("Alertes file", n_alerte)
    c4.metric("F1 modèle", info["metriques"].get("f1", "—"))

    st.subheader("Scores de la file")
    st.altair_chart(score_chart(queue, seuil), use_container_width=True)

    st.subheader("Top risques")
    top = queue.head(10)[["id", "montant", "heure", "score", "label_reel"]].copy()
    top["niveau"] = top["score"].map(lambda s: risk_level(float(s), seuil))
    top = top.rename(
        columns={
            "id": "Transaction",
            "montant": "Montant (€)",
            "heure": "Heure",
            "score": "Score",
            "label_reel": "Label réel",
            "niveau": "Niveau",
        }
    )
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0%"),
            "Montant (€)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    g1, g2 = st.columns(2)
    with g1:
        img = FIGURES_DIR / "classes.png"
        if img.exists():
            st.image(str(img), caption="Déséquilibre des classes", use_container_width=True)
    with g2:
        img = FIGURES_DIR / "montants.png"
        if img.exists():
            st.image(str(img), caption="Montants", use_container_width=True)


elif page == "File d'attente":
    st.header("File d'attente")
    st.write("Choisissez une transaction, puis consultez le dossier à droite.")

    c1, c2 = st.columns(2)
    with c1:
        filtre = st.radio(
            "Filtre",
            ["Toutes", "Alertes", "Surveillance", "Non revues"],
            horizontal=True,
        )
    with c2:
        tri = st.selectbox("Tri", ["Score ↓", "Montant ↓", "ID"])

    view = queue.copy()
    view["niveau"] = view["score"].map(lambda s: risk_level(float(s), seuil))
    if filtre == "Alertes":
        view = view[view["score"] >= seuil]
    elif filtre == "Surveillance":
        view = view[view["niveau"] == "Modéré"]
    elif filtre == "Non revues":
        view = view[~view["id"].isin(st.session_state.reviewed)]

    if tri == "Montant ↓":
        view = view.sort_values("montant", ascending=False)
    elif tri == "ID":
        view = view.sort_values("id")
    else:
        view = view.sort_values("score", ascending=False)
    view = view.reset_index(drop=True)

    if view.empty:
        st.warning("Aucun dossier pour ce filtre.")
        st.stop()

    options = view["id"].tolist()
    default_idx = 0
    if st.session_state.selected_tx in options:
        default_idx = options.index(st.session_state.selected_tx)

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        choix = st.selectbox("Transaction", options, index=default_idx)
        st.session_state.selected_tx = choix

        table = view[["id", "montant", "heure", "score", "niveau", "label_reel"]].rename(
            columns={
                "id": "Transaction",
                "montant": "Montant (€)",
                "heure": "Heure",
                "score": "Score",
                "niveau": "Niveau",
                "label_reel": "Label réel",
            }
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0%"),
                "Montant (€)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with right:
        row = view[view["id"] == st.session_state.selected_tx].iloc[0]
        st.subheader(f"Dossier {row['id']}")
        st.write(f"**Montant :** {row['montant']:.2f} €")
        st.write(f"**Heure :** {row['heure']}")
        st.write(f"**Time :** {int(row['temps_s'])} s")
        show_verdict(float(row["score"]), seuil, str(row["label_reel"]))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Valider via API", type="primary", use_container_width=True):
                data = call_api({c: float(row[c]) for c in FEATURES})
                if data:
                    st.session_state["last_api"] = data
                    st.success("Confirmé par l'API")
        with b2:
            if st.button("Marquer revue", use_container_width=True):
                st.session_state.reviewed.add(row["id"])
                st.success(f"{row['id']} marquée revue")

        if "last_api" in st.session_state:
            with st.expander("Réponse API"):
                st.json(st.session_state["last_api"])
        with st.expander("Variables PCA V1–V28"):
            st.dataframe(
                pd.DataFrame(
                    {"variable": PCA_FEATURES, "valeur": [float(row[c]) for c in PCA_FEATURES]}
                ),
                hide_index=True,
                use_container_width=True,
                height=260,
            )


elif page == "Analyse manuelle":
    st.header("Analyse manuelle")
    st.write("Saisie des caractéristiques + appel API.")

    if "form_defaults" not in st.session_state:
        base = queue[queue["Class"] == 0].iloc[0]
        st.session_state.form_defaults = {c: float(base[c]) for c in FEATURES}

    p1, p2, p3 = st.columns(3)
    if p1.button("Profil légitime", use_container_width=True):
        base = queue[queue["Class"] == 0].iloc[0]
        st.session_state.form_defaults = {c: float(base[c]) for c in FEATURES}
        st.session_state.manual_result = None
        st.rerun()
    if p2.button("Profil fraude", use_container_width=True):
        base = queue[queue["Class"] == 1].iloc[0]
        st.session_state.form_defaults = {c: float(base[c]) for c in FEATURES}
        st.session_state.manual_result = None
        st.rerun()
    pick = p3.selectbox("Reprendre une TX", ["—"] + queue["id"].tolist())
    if pick != "—" and st.button("Charger cette TX", use_container_width=True):
        base = queue[queue["id"] == pick].iloc[0]
        st.session_state.form_defaults = {c: float(base[c]) for c in FEATURES}
        st.session_state.manual_result = None
        st.rerun()

    defaults = st.session_state.form_defaults
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        with st.form("form_analyse"):
            a1, a2 = st.columns(2)
            amount = a1.number_input(
                "Montant (€)", 0.0, 30000.0, float(defaults.get("Amount", 50.0)), 1.0
            )
            time_v = a2.number_input(
                "Time (secondes)", 0.0, 200000.0, float(defaults.get("Time", 0.0)), 1.0
            )
            with st.expander("Variables PCA V1–V28"):
                for i, name in enumerate(PCA_FEATURES):
                    if i % 4 == 0:
                        cols = st.columns(4)
                    with cols[i % 4]:
                        st.number_input(
                            name,
                            value=float(defaults.get(name, 0.0)),
                            format="%.4f",
                            key=f"inp_{name}",
                        )
            go = st.form_submit_button("Analyser", type="primary", use_container_width=True)

        if go:
            payload = {"Time": float(time_v), "Amount": float(amount)}
            for name in PCA_FEATURES:
                payload[name] = float(
                    st.session_state.get(f"inp_{name}", defaults.get(name, 0.0))
                )
            st.session_state.form_defaults = payload
            data = call_api(payload)
            if data:
                st.session_state.manual_result = {"data": data, "payload": payload}

    with right:
        st.subheader("Verdict")
        res = st.session_state.manual_result
        if not res:
            st.info("Le résultat apparaîtra ici après analyse.")
        else:
            show_verdict(float(res["data"]["probabilite_fraude"]), seuil)
            st.write(
                f"Modèle `{res['data'].get('modele')}` · version `{res['data'].get('model_version')}`"
            )
            with st.expander("Détails"):
                st.json(res)


else:
    st.header("Performance modèle")
    st.write("Comparaison des approches et modèle retenu.")
    m = info.get("metriques", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retenu", info.get("modele_retenu", "?"))
    c2.metric("F1", m.get("f1", "—"))
    c3.metric("Rappel", m.get("rappel", "—"))
    c4.metric("PR-AUC", m.get("pr_auc", "—"))
    st.info(info.get("justification", ""))
    if COMPARISON_PATH.exists():
        st.dataframe(pd.read_csv(COMPARISON_PATH), use_container_width=True, hide_index=True)
    g1, g2 = st.columns(2)
    with g1:
        img = FIGURES_DIR / "comparaison.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
    with g2:
        img = FIGURES_DIR / "confusion.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
