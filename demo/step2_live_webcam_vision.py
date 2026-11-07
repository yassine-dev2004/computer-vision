"""
Étape 2 : Démonstration Temps Réel Webcam (SignNet CNN + TTS + HUD)
===================================================================
SignWave AI — Inférence temps réel accélérée sur GPU avec interface moderne.

Placez votre main dans le cadre vert/cyan pour prédire en direct.
Raccourcis :
  - [ESPACE] : Insère un espace
  - [RETOUR ARRIÈRE] : Efface la dernière lettre
  - [S] : Lit la phrase à voix haute (Synthèse Vocale)
  - [C] : Efface toute la phrase
  - [Q] ou [ECHAP] : Quitter
"""

import os
import sys
import json
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Import des modules internes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.step1_train_dataset_cnn import SignNet
from demo.src.speller import SignSpeller
from demo.src.tts_engine import TTSEngine


def draw_hud(frame, fps, pred_char, confidence, top3_preds, hold_progress, speller_text, current_word, roi_coords):
    """Dessine l'interface tête-haute (HUD) complète."""
    h, w, _ = frame.shape
    rx, ry, rw, rh = roi_coords

    # 1. Cadre de visée de la main (ROI)
    box_color = (6, 182, 212) if confidence > 0.75 else (99, 102, 241)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
    # Coins décoratifs technologiques
    corner_len = 20
    # Haut gauche
    cv2.line(frame, (rx, ry), (rx + corner_len, ry), (16, 185, 129), 4)
    cv2.line(frame, (rx, ry), (rx, ry + corner_len), (16, 185, 129), 4)
    # Bas droite
    cv2.line(frame, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), (16, 185, 129), 4)
    cv2.line(frame, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), (16, 185, 129), 4)

    cv2.putText(frame, "ZONE DE GESTE", (rx + 5, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # 2. Bandeau supérieur (Titre & FPS)
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 23, 42), -1)
    cv2.line(frame, (0, 55), (w, 55), (99, 102, 241), 2)
    cv2.putText(frame, "SignWave AI - Traducteur Temps Reel (2 IRM)", (25, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (6, 182, 212), 2)

    # 3. Panneau latéral gauche (Prédictions & Probabilités)
    panel_w = 260
    cv2.rectangle(frame, (20, 75), (20 + panel_w, 360), (15, 23, 42), -1)
    cv2.rectangle(frame, (20, 75), (20 + panel_w, 360), (255, 255, 255), 1)

    cv2.putText(frame, "SIGNE DETECTE", (35, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

    char_display = str(pred_char) if pred_char and confidence > 0.4 else "--"
    cv2.putText(frame, char_display, (110, 185),
                cv2.FONT_HERSHEY_DUPLEX, 2.6, (6, 182, 212), 4)

    # Jauge de confiance
    cv2.putText(frame, f"Confiance : {confidence*100:.1f}%", (35, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    bar_x, bar_y, bar_w, bar_h = 35, 230, 200, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * confidence)
    col = (16, 185, 129) if confidence > 0.75 else ((245, 158, 11) if confidence > 0.5 else (244, 63, 94))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), col, -1)

    # Progression du maintien (Hold timer)
    if hold_progress > 0:
        cv2.putText(frame, f"Validation : {int(hold_progress*100)}%", (35, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212), 1)
        hold_w = int(bar_w * hold_progress)
        cv2.rectangle(frame, (bar_x, 265), (bar_x + hold_w, 270), (6, 182, 212), -1)

    # Top 3 des probabilités
    cv2.putText(frame, "Top 3 Prédictions :", (35, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
    for i, (c, p) in enumerate(top3_preds):
        cv2.putText(frame, f" {i+1}. {c}  ({p*100:.1f}%)", (40, 318 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # 4. Bandeau inférieur (Phrase & Mots)
    cv2.rectangle(frame, (0, h - 90), (w, h), (15, 23, 42), -1)
    cv2.line(frame, (0, h - 90), (w, h - 90), (6, 182, 212), 2)

    cv2.putText(frame, "PHRASE :", (25, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (148, 163, 184), 1)
    display_phrase = speller_text if speller_text else "[Placez votre main dans le cadre pour epeler...]"
    cv2.putText(frame, display_phrase, (130, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    cv2.putText(frame, "[ESPACE] Espace  |  [RETOUR] Effacer  |  [S] Synthèse Vocale  |  [C] Vider  |  [Q] Quitter",
                (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "sign_classifier.pth")
    classes_path = os.path.join(base_dir, "models", "classes.json")

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("[ERREUR] Le modèle n'est pas encore entraîné.")
        print("Exécutez d'abord : python demo/step1_train_dataset_cnn.py")
        return

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement de SignNet sur {device}...")
    model = SignNet(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transformation d'inférence
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    speller = SignSpeller(hold_threshold=1.2, min_confidence=0.70)
    tts = TTSEngine()

    print("\nConnexion à la webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        print("Vérifiez les permissions de votre caméra sous Windows.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 65)
    print("  SignWave AI Démo Webcam démarrée !")
    print("  Positionnez votre main dans le cadre pour traduire.")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calcul du FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = curr_time

            # Définition de la boîte ROI au centre-droit
            box_size = int(min(w, h) * 0.48)
            rx = int(w * 0.55)
            ry = int((h - box_size) / 2)
            roi_coords = (rx, ry, box_size, box_size)

            # Extraction du crop ROI
            crop = frame[ry:ry + box_size, rx:rx + box_size]

            pred_char = None
            confidence = 0.0
            top3 = []
            hold_progress = 0.0

            if crop.size > 0:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)[0]

                    top_probs, top_indices = torch.topk(probs, k=3)
                    top3 = [(classes[idx.item()], float(prob.item())) for prob, idx in zip(top_probs, top_indices)]

                    pred_char = top3[0][0]
                    confidence = top3[0][1]

                # Mise à jour du speller
                res = speller.update(pred_char, confidence)
                hold_progress = res["progress"]

                if res["letter_added"]:
                    added = res["letter_added"]
                    if added not in [" ", "<DEL>"]:
                        tts.speak(added, min_interval_seconds=0.4)

            # Dessin de l'interface
            draw_hud(
                frame=frame,
                fps=fps,
                pred_char=pred_char,
                confidence=confidence,
                top3_preds=top3,
                hold_progress=hold_progress,
                speller_text=speller.get_full_text(),
                current_word=speller.current_word,
                roi_coords=roi_coords
            )

            cv2.imshow("SignWave AI - Live Camera Translator (2 IRM)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == ord(' '):
                speller._commit_char(" ")
            elif key in [8, 127]:
                speller._commit_char("DEL")
            elif key in [ord('s'), ord('S')]:
                phrase = speller.get_full_text()
                if phrase:
                    print(f"[TTS Audio] : {phrase}")
                    tts.speak(phrase, min_interval_seconds=0.0)
            elif key in [ord('c'), ord('C')]:
                speller.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("\nSession caméra terminée.")


if __name__ == "__main__":
    main()
