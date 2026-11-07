# 🤟 SignWave AI — Dossier Démonstration Complète (2 IRM)

Ce sous-dossier `demo/` contient l'ensemble du projet **SignWave AI** fonctionnel de bout en bout, entraîné sur votre dossier `DATASET/` (36 classes : chiffres 0 à 9 et lettres A à Z).

---

## 📁 Structure du Dossier `demo/`

```text
demo/
│
├── step1_train_dataset_cnn.py      # Entraînement Deep Learning (PyTorch GPU/CPU) -> 93.9% de précision
├── step2_live_webcam_vision.py     # Démo webcam temps réel avec HUD, épellation & synthèse vocale
├── step3_mediapipe_landmarks.py    # Démo tracking squelette 3D MediaPipe & normalisation géométrique
├── step4_dynamic_words.py          # Démo mots dynamiques par réseau récurrent LSTM (30 frames)
├── app_dashboard.py                # Dashboard interactif Web Streamlit
│
├── models/                         # Modèles entraînés et métriques
│   ├── sign_classifier.pth         # Poids du modèle PyTorch SignNet (8.6 Mo)
│   ├── classes.json                # Liste des 36 classes (0-9, A-Z)
│   └── confusion_matrix.png        # Matrice de confusion générée
│
└── src/                            # Modules réutilisables
    ├── vision_tracker.py           # Tracker MediaPipe Hands
    ├── normalizer.py               # Formule de normalisation vectorielle P'_i
    ├── speller.py                  # Moteur d'assemblage de mots & anti-rebond
    └── tts_engine.py               # Synthèse vocale multithreadée asynchrone (pyttsx3)
```

---

## 🚀 Comment Lancer les Différentes Démos

### 1. Démo Principale : Traducteur Webcam en Temps Réel
Lancez le traducteur direct avec le modèle entraîné sur votre GPU NVIDIA RTX :
```bash
python demo/step2_live_webcam_vision.py
```
* **Utilisation :** Placez votre main dans le cadre cyan.
* **Raccourcis :**
  - Maintenir un signe 1.2s pour l'écrire dans la phrase.
  - `[ESPACE]` pour ajouter un espace.
  - `[RETOUR ARRIÈRE]` pour effacer une lettre.
  - `[S]` pour prononcer la phrase à voix haute via la synthèse vocale.
  - `[C]` pour vider le texte.
  - `[Q]` pour quitter.

---

### 2. Démo Squelette 3D MediaPipe (21 Landmarks)
Pour montrer au jury l'extraction géométrique et l'invariance spatiale :
```bash
python demo/step3_mediapipe_landmarks.py
```

---

### 3. Démo Mots Dynamiques (Modèle Séquentiel LSTM)
Pour illustrer la détection temporelle sur fenêtres glissantes de 30 frames :
```bash
python demo/step4_dynamic_words.py
```

---

### 4. Dashboard Web Interactif (Streamlit)
Pour présenter les graphiques, la matrice de confusion et les benchmarks au jury :
```bash
streamlit run demo/app_dashboard.py
```

---

### 5. Ré-entraîner le Modèle (Optionnel)
Pour ré-entraîner le réseau convolutif depuis zéro sur `DATASET/` :
```bash
python demo/step1_train_dataset_cnn.py
```
*(Entraînement complet en ~70 secondes sur votre GPU NVIDIA RTX 2050)*.
