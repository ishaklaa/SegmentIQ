from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Configuration de la page ----------
st.set_page_config(page_title="SegmentIQ", page_icon="🎯", layout="centered")

# ---------- Chargement du modèle ----------
# Les fichiers .joblib sont dans le dossier notebooks/ (à côté de ce fichier : SegmentIQ/app.py)
MODEL_DIR = Path(__file__).parent / "notebooks"


@st.cache_resource
def charger_modele():
    modele = joblib.load(MODEL_DIR / "modele_rf_segmentation.joblib")
    noms = joblib.load(MODEL_DIR / "noms_segments.joblib")
    return modele, noms


try:
    modele, noms_segments = charger_modele()
except FileNotFoundError:
    st.error(
        "Modèle introuvable. Exécutez d'abord le notebook `Segmentation_Client.ipynb` "
        "pour générer `modele_rf_segmentation.joblib` et `noms_segments.joblib` "
        "dans le dossier `notebooks/`."
    )
    st.stop()

# ---------- Titre ----------
st.title("🎯 SegmentIQ")
st.write(
    "Renseignez les informations d'un client pour découvrir à quel **segment** "
    "il appartient."
)

# ---------- Formulaire de saisie ----------
st.subheader("Informations du client")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Récence (jours)",
        min_value=0,
        value=30,
        step=1,
        help="Nombre de jours écoulés depuis le dernier achat.",
    )

with col2:
    frequency = st.number_input(
        "Fréquence (commandes)",
        min_value=1,
        value=5,
        step=1,
        help="Nombre total de commandes passées par le client.",
    )

with col3:
    monetary = st.number_input(
        "Montant total dépensé",
        min_value=0.0,
        value=1000.0,
        step=50.0,
        help="Somme totale dépensée par le client.",
    )

# ---------- Prédiction ----------
if st.button("Prédire le segment", type="primary", use_container_width=True):
    nouveau_client = pd.DataFrame(
        {"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]}
    )

    cluster = int(modele.predict(nouveau_client)[0])
    nom_segment = noms_segments.get(cluster, f"Segment {cluster}")

    st.divider()
    st.subheader("Résultat")

    st.success(f"Ce client appartient au segment : **{nom_segment}**")

    r1, r2 = st.columns(2)
    r1.metric("Cluster prédit", cluster)
    r2.metric("Nom du segment", nom_segment)

    st.subheader("Caractéristiques RFM saisies")
    c1, c2, c3 = st.columns(3)
    c1.metric("Récence", f"{recency} jours")
    c2.metric("Fréquence", f"{frequency} commandes")
    c3.metric("Montant total", f"{monetary:,.2f}")

    # Indice de confiance du modèle
    probas = modele.predict_proba(nouveau_client)[0]
    with st.expander("Voir la confiance du modèle"):
        tableau = pd.DataFrame(
            {
                "Segment": [noms_segments.get(int(c), f"Segment {c}") for c in modele.classes_],
                "Probabilité": [f"{p:.1%}" for p in probas],
            }
        )
        st.table(tableau)