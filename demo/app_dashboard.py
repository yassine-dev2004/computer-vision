"""
Application Web Interactive — Dashboard SignWave AI
===================================================
Interface Streamlit de démonstration pour le jury (2 IRM).

Lancement :
  streamlit run demo/app_dashboard.py
"""

import os
import sys
import json
import random
from PIL import Image
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="SignWave AI — Dashboard",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS Cyberpunk / Modern Dark
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06B6D4 0%, #6366F1 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .badge-2irm {
        background: linear-gradient(135deg, #06B6D4, #6366F1);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #06B6D4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "..", "DATASET")
models_dir = os.path.join(base_dir, "models")
cm_path = os.path.join(models_dir, "confusion_matrix.png")
classes_path = os.path.join(models_dir, "classes.json")

# En-tête
col_t1, col_t2 = st.columns([0.8, 0.2])
with col_t1:
    st.markdown('<h1 class="main-title">SignWave AI</h1>', unsafe_allow_html=True)
    st.markdown('**Traducteur de Langue des Signes en Temps Réel** &nbsp;|&nbsp; <span class="badge-2irm">2 IRM</span>', unsafe_allow_html=True)
with col_t2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'><a href='presentation_signwave_ai.html' target='_blank' style='text-decoration:none; background:#1E293B; color:#38BDF8; padding:8px 16px; border-radius:8px; border:1px solid #06B6D4;'>📊 Ouvrir Slides</a></div>", unsafe_allow_html=True)

st.write("")

# Navigation Sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Modules :", [
    "🏠 Vue d'ensemble & Architecture",
    "📊 Benchmarks & Matrice de Confusion",
    "📁 Explorateur du Dataset (36 Classes)",
    "🎙️ Simulateur Synthèse Vocale (TTS)",
    "🚀 Guide de Lancement Webcam"
])

# 1. ACCUEIL
if menu == "🏠 Vue d'ensemble & Architecture":
    st.subheader("Architecture & Métriques de Performance")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card"><div class="metric-val">93.9%</div><div class="metric-label">Précision Globale (Val)</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card"><div class="metric-val">36</div><div class="metric-label">Classes (0-9 & A-Z)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><div class="metric-val">&lt; 3 ms</div><div class="metric-label">Latence par Image (GPU)</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card"><div class="metric-val">&gt; 45</div><div class="metric-label">Images par Seconde (FPS)</div></div>""", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Pipeline de Traitement")
    st.markdown("""
    ```text
    [ Flux Webcam (OpenCV) ] ──> [ Normalisation Géométrique / ROI ] 
                                           │
                                           ▼
                             [ Réseau Convolutif SignNet ] ──> (Inférence GPU < 3ms)
                                           │
                                           ▼
                                [ Filtre Anti-Rebond ] ──> (Maintien > 1.2s)
                                           │
                                           ▼
                             [ Synthèse Vocale Audio (TTS) ]
    ```
    """)

# 2. BENCHMARKS
elif menu == "📊 Benchmarks & Matrice de Confusion":
    st.subheader("Évaluation du Modèle & Matrice de Confusion")

    if os.path.exists(cm_path):
        st.image(cm_path, caption="Matrice de confusion sur les 36 classes (Validation Set)", use_column_width=True)
    else:
        st.info("La matrice de confusion est en cours de génération...")

    st.write("---")
    st.subheader("Benchmark Matériel")
    st.table({
        "Composant": ["GPU Utilisé", "Architecture Modèle", "Taille du Poids (.pth)", "Nombre d'époques", "Temps d'inférence"],
        "Spécification": ["NVIDIA GeForce RTX 2050 (CUDA)", "SignNet (4 Blocs Conv2d + BatchNorm)", "8.6 Mo", "20 Époques", "2.1 ms"]
    })

# 3. EXPLORATEUR DATASET
elif menu == "📁 Explorateur du Dataset (36 Classes)":
    st.subheader("Explorateur Interactif des Données (DATASET/)")

    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = sorted(os.listdir(dataset_dir)) if os.path.exists(dataset_dir) else []

    selected_class = st.selectbox("Sélectionnez une classe :", classes)

    if selected_class and os.path.exists(dataset_dir):
        class_folder = os.path.join(dataset_dir, selected_class)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        st.write(f"Nombre d'images pour la classe **'{selected_class}'** : {len(images)}")

        cols = st.columns(6)
        for i, img_name in enumerate(images[:12]):
            with cols[i % 6]:
                img_p = os.path.join(class_folder, img_name)
                st.image(Image.open(img_p), caption=f"{selected_class} (#{i+1})", use_column_width=True)

# 4. SIMULATEUR TTS
elif menu == "🎙️ Simulateur Synthèse Vocale (TTS)":
    st.subheader("Simulateur Audio & Épellation")
    test_text = st.text_input("Tapez un mot ou une phrase à prononcer :", "BONJOUR 2 IRM")

    if st.button("🔊 Tester la Synthèse Vocale"):
        from demo.src.tts_engine import TTSEngine
        tts = TTSEngine()
        tts.speak(test_text, min_interval_seconds=0.0)
        st.success(f"Lecture audio envoyée : '{test_text}'")

# 5. GUIDE DE LANCEMENT
elif menu == "🚀 Guide de Lancement Webcam":
    st.subheader("Comment Lancer la Démonstration Webcam en Direct")
    st.markdown("""
    Pour lancer le traducteur en temps réel sur votre caméra, ouvrez un terminal dans ce dossier et lancez :

    ```bash
    python demo/step2_live_webcam_vision.py
    ```

    ### Commandes au Clavier :
    - **Positionnez votre main** dans le cadre vert/cyan pour voir la prédiction en direct.
    - **Maintenez le signe 1.2 seconde** pour valider la lettre dans la phrase.
    - **[ESPACE]** : Insère un espace.
    - **[RETOUR ARRIÈRE]** : Efface le dernier caractère.
    - **[S]** : Prononce la phrase complète à voix haute.
    - **[C]** : Efface toute la phrase.
    - **[Q]** : Quitter la caméra.
    """)

