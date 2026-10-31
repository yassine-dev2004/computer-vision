# 🤟 SignWave AI — Traducteur de Langue des Signes en Temps Réel

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%26%20Pose-teal.svg)](https://developers.google.com/mediapipe)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%2F%20Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-30%2B%20FPS-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Projet d'Excellence en Vision par Ordinateur & Deep Learning**  
> **Filière :** 2 IRM (Ingénierie Réseaux & Multimédia)  
> **Module :** Computer Vision & Intelligence Artificielle

---

## 📖 À Propos du Projet

**SignWave AI** est une passerelle de communication bidirectionnelle et non-intrusive conçue pour briser la barrière entre les personnes sourdes ou malentendantes et les entendants. 

Grâce à une webcam standard (aucun capteur matériel coûteux requis), le système capture les gestes de la main et du corps, extrait les repères articulaires 3D en temps réel, les normalise géométriquement et prédit les lettres (alphabet statique) ou les mots (signes dynamiques séquentiels) pour les restituer sous forme de texte et de voix synthétisée.

---

## 🌟 Fonctionnalités Clés

- **Tracking Haute Vitesse (30+ FPS) :** Détection instantanée des 21 repères 3D des mains via **Google MediaPipe**.
- **Invariance Arrière-Plan & Éclairage :** Représentation squelettique vectorielle pure (évite le surapprentissage sur les pixels ou l'environnement).
- **Normalisation Géométrique Mathématique :**
  $$\mathbf{P}'_i = \frac{\mathbf{P}_i - \mathbf{P}_0}{\|\mathbf{P}_9 - \mathbf{P}_0\|}$$
  Rend les coordonnées invariantes à l'échelle, à l'éloignement et à la position dans le champ de la caméra.
- **Approche Hybride Statique / Dynamique :**
  - *Statique (Alphabet ASL A–Z) :* Inférence ultra-rapide par Random Forest / MLP.
  - *Dynamique (Mots complets) :* Analyse spatio-temporelle via réseau de neurones récurrent **LSTM** sur fenêtres glissantes de 30 frames.
- **Passerelle Vocale :** Synthèse audio instantanée locale (**pyttsx3**) et sous-titrage automatique.
- **Présentation Interactive Embarquée :** Support de présentation HTML5 interactif avec rendu 3D de squelette de main (`presentation_signwave_ai.html`).

---

## 🏛 Architecture du Pipeline

```text
[ Caméra OpenCV ] ──(30 FPS)──> [ MediaPipe Hands (21 Landmarks 3D) ]
                                          │
                                          ▼
                               [ Normalisation Géométrique ]
                               (Centrage Poignet P0 & Échelle P9)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            [ Alphabet Statique ]                    [ Signes Dynamiques ]
             (Random Forest / MLP)                  (Fenêtre 30 frames + LSTM)
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼
                               [ Filtre Anti-Rebond ]
                                 (Confiance > 85%)
                                          │
                                          ▼
                         [ Synthèse Vocale (TTS) & UI ]
```

---

## 📊 Datasets Utilisés

1. **Alphabet Statique (A–Z) :**
   - [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87 000 images, 29 classes).
   - [Sign Language MNIST](https://www.kaggle.com/datasets/datamarket/sign-language-mnist) (format léger pour prototypage).
2. **Mots & Signes Dynamiques :**
   - [WLASL (Word-Level American Sign Language)](https://github.com/dxli92/WLASL) — Sous-ensemble WLASL-100.
   - [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs).
   - **Dataset Personnalisé :** Outil de capture interactif pour enregistrer des gestes sur mesure.

---

## 📂 Structure du Répertoire

```text
computer-vision/
│
├── presentation_signwave_ai.html  # Présentation interactive du projet (Slide Deck HTML5)
├── README.md                      # Documentation du projet
├── .gitignore                     # Fichiers exclus du versionnement
│
├── data/                          # Données d'entraînement et repères extraits
│   ├── raw/                       # Images/vidéos brutes (non versionnées)
│   └── processed/                 # Fichiers vectoriels (.npy / .csv)
│
├── src/                           # Code source du pipeline
│   ├── vision/                    # Extraction MediaPipe
│   ├── preprocessing/            # Normalisation et fenêtrage
│   ├── models/                    # Architectures MLP & LSTM
│   └── audio/                     # Moteur TTS
│
└── app/                           # Interface de démo (Streamlit / OpenCV GUI)
```

---

## 🚀 Démarrage Rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/yassine-dev2004/computer-vision.git
cd computer-vision
```

### 2. Créer l'environnement virtuel & installer les dépendances
```bash
python -m venv venv
# Windows :
.\venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate

pip install opencv-python mediapipe numpy torch scikit-learn pyttsx3 streamlit
```

### 3. Lancer la présentation
Ouvrez simplement le fichier `presentation_signwave_ai.html` dans n'importe quel navigateur moderne (touches `◄` / `►` pour naviguer, `O` pour la grille des slides, `F` pour le plein écran).

---

## 👥 Auteur
- **Yassine Mezghanni** ([@yassine-dev2004](https://github.com/yassine-dev2004))
- Étudiant en **2 IRM** (Ingénierie Réseaux & Multimédia)
