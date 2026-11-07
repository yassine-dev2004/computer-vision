"""
Étape 3 : Application Temps Réel Webcam (Inférence, Épellation & TTS)
====================================================================
SignWave AI — Démonstration interactive avec interface HUD avancée.

Raccourcis clavier :
  - [ESPACE] : Insère un espace dans la phrase
  - [RETOUR ARRIÈRE] : Supprime le dernier caractère
  - [S] : Prononce la phrase complète via synthèse vocale
  - [C] : Réinitialise le texte en cours
  - [Q] ou [ECHAP] : Quitter
"""

import os
import sys
import time
import pickle
import cv2
import numpy as np

# Import des modules internes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.src.vision_tracker import HandTracker
from demo.src.normalizer import LandmarkNormalizer
from demo.src.speller import SignSpeller
from demo.src.tts_engine import TTSEngine


def draw_hud(frame, fps, pred_char, confidence, hold_progress, speller_text, current_word):
    """Dessine une interface tête-haute (HUD) moderne et cyberpunk."""
    h, w, _ = frame.shape

    # 1. Barre supérieure (Titre & FPS)
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 23, 42), -1)
    cv2.line(frame, (0, 50), (w, 50), (241, 102, 99), 2)
    cv2.putText(frame, "SignWave AI - Traducteur Temps Reel (2 IRM)", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (6, 182, 212), 2)

    # 2. Carte de Prédiction (Haut Gauche)
    cv2.rectangle(frame, (20, 65), (220, 240), (15, 23, 42), -1)
    cv2.rectangle(frame, (20, 65), (220, 240), (255, 255, 255), 1)

    cv2.putText(frame, "SIGNE DETECTE", (35, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    char_display = str(pred_char) if pred_char else "--"
    cv2.putText(frame, char_display, (85, 170),
                cv2.FONT_HERSHEY_DUPLEX, 2.4, (6, 182, 212), 4)

    # Jauge de confiance
    cv2.putText(frame, f"Confiance: {confidence*100:.1f}%", (35, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    bar_x, bar_y, bar_w, bar_h = 35, 210, 150, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * confidence)
    color = (16, 185, 129) if confidence > 0.8 else ((245, 158, 11) if confidence > 0.6 else (244, 63, 94))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # 3. Barre de progression du maintien (Hold Timer)
    if hold_progress > 0:
        cv2.putText(frame, "Validation...", (35, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (6, 182, 212), 1)
        hold_w = int(bar_w * hold_progress)
        cv2.rectangle(frame, (bar_x, 242), (bar_x + hold_w, 246), (6, 182, 212), -1)

    # 4. Bandeau inférieur (Mot & Phrase assemblée)
    cv2.rectangle(frame, (0, h - 90), (w, h), (15, 23, 42), -1)
    cv2.line(frame, (0, h - 90), (w, h - 90), (6, 182, 212), 2)

    cv2.putText(frame, "PHRASE :", (20, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (148, 163, 184), 1)
    cv2.putText(frame, speller_text if speller_text else "[En attente de signes...]", (110, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Raccourcis: [ESPACE] Espace | [RETOUR] Effacer | [S] Parler | [C] Vider | [Q] Quitter",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "static_classifier.pkl")
    encoder_path = os.path.join(base_dir, "models", "label_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"[ERREUR] Le modèle n'est pas encore entraîné.")
        print(f"Exécutez d'abord : python demo/step2_train_and_evaluate.py")
        return

    print("Chargement du modèle...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    tracker = HandTracker()
    normalizer = LandmarkNormalizer()
    speller = SignSpeller(hold_threshold=1.0, min_confidence=0.75)
    tts = TTSEngine()

    print("\nInitialisation de la caméra...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tentative avec caméra index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'accéder à la webcam.")
        print("Vérifiez les permissions de votre caméra sous Windows.")
        return

    # Configuration résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 65)
    print("  SignWave AI Démo Temps Réel démarrée !")
    print("  Positionnez votre main devant la caméra.")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Miroir horizontal pour effet miroir naturel
            frame = cv2.flip(frame, 1)

            # Extraction MediaPipe
            landmarks_list, frame = tracker.process_frame(frame, draw_styled=True)

            pred_char = None
            confidence = 0.0
            hold_progress = 0.0

            if landmarks_list:
                landmarks = landmarks_list[0].landmark
                features = normalizer.normalize(landmarks)

                # Inférence avec probabilités
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([features])[0]
                    best_idx = np.argmax(probs)
                    confidence = float(probs[best_idx])
                    pred_char = label_encoder.inverse_transform([best_idx])[0]
                else:
                    pred_idx = model.predict([features])[0]
                    confidence = 1.0
                    pred_char = label_encoder.inverse_transform([pred_idx])[0]

                # Mise à jour du speller
                res = speller.update(pred_char, confidence)
                hold_progress = res["progress"]

                # Si une lettre est validée, on peut prononcer la lettre ou le mot
                if res["letter_added"]:
                    letter = res["letter_added"]
                    if letter not in [" ", "<DEL>"]:
                        tts.speak(letter, min_interval_seconds=0.5)

            # Dessin de l'interface graphique
            draw_hud(
                frame=frame,
                fps=tracker.get_fps(),
                pred_char=pred_char,
                confidence=confidence,
                hold_progress=hold_progress,
                speller_text=speller.get_full_text(),
                current_word=speller.current_word
            )

            cv2.imshow("SignWave AI - Live Camera Translator", frame)

            # Gestion des touches clavier
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # Q ou Echap
                break
            elif key == ord(' '):  # Espace
                speller._commit_char(" ")
            elif key in [8, 127]:  # Backspace
                speller._commit_char("DEL")
            elif key in [ord('s'), ord('S')]:  # Synthèse vocale de la phrase
                full_phrase = speller.get_full_text()
                if full_phrase:
                    print(f"[TTS] Lecture : {full_phrase}")
                    tts.speak(full_phrase, min_interval_seconds=0.0)
            elif key in [ord('c'), ord('C')]:  # Vider
                speller.clear()

    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("\nSession terminée.")


if __name__ == "__main__":
    main()
