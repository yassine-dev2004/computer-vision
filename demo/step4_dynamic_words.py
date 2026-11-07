"""
Étape 4 : Reconnaissance des Gestes Dynamiques (Modèle Séquentiel LSTM)
=======================================================================
SignWave AI — Traitement spatio-temporel sur fenêtres glissantes de 30 frames.

Architecture :
  Input (30 frames x 63 features) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(Softmax)
Vocabulaire dynamique :
  ["BONJOUR", "MERCI", "AIDE", "S'IL VOUS PLAIT", "OUI", "NON", "REPOS"]
"""

import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.src.vision_tracker import HandTracker
from demo.src.normalizer import LandmarkNormalizer
from demo.src.tts_engine import TTSEngine


class SignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, num_classes=7):
        super(SignLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, 32, batch_first=True, num_layers=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, (hn, _) = self.lstm2(out)
        out = self.fc(hn[-1])
        return out


def main():
    print("=" * 65)
    print("  SignWave AI — Mots Dynamiques & LSTM Temps Réel")
    print("=" * 65)

    vocabulary = ["BONJOUR", "MERCI", "AIDE", "S'IL VOUS PLAIT", "OUI", "NON", "REPOS"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignLSTM(input_size=63, hidden_size=64, num_classes=len(vocabulary)).to(device)
    model.eval()

    tracker = HandTracker()
    normalizer = LandmarkNormalizer()
    tts = TTSEngine()

    sequence_length = 30
    frame_buffer = deque(maxlen=sequence_length)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_prediction = "REPOS"
    last_spoken_time = 0.0

    print("\nFenêtre glissante prête : 30 frames par séquence temporelle.")
    print("Appuyez sur [Q] pour quitter.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            landmarks_list, annotated = tracker.process_frame(frame, draw_styled=True)
            fps = tracker.get_fps()

            if landmarks_list:
                landmarks = landmarks_list[0].landmark
                features = normalizer.normalize(landmarks)
                frame_buffer.append(features)
            else:
                # Vecteur neutre si pas de main
                frame_buffer.append(np.zeros(63, dtype=np.float32))

            # Inférence par fenêtre glissante dès qu'on a 30 frames
            current_pred = "EN ATTENTE..."
            confidence = 0.0

            if len(frame_buffer) == sequence_length:
                seq_tensor = torch.tensor(np.array([list(frame_buffer)]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(seq_tensor)
                    probs = F.softmax(logits, dim=1)[0]
                    best_idx = torch.argmax(probs).item()
                    confidence = probs[best_idx].item()
                    current_pred = vocabulary[best_idx]

                # Filtre anti-faux positifs et validation de prononciation
                now = time.time()
                if current_pred != "REPOS" and confidence > 0.80 and (now - last_spoken_time) > 2.0:
                    tts.speak(current_pred)
                    last_spoken_time = now
                    last_prediction = current_pred

            # HUD
            cv2.rectangle(annotated, (0, 0), (w, 55), (15, 23, 42), -1)
            cv2.line(annotated, (0, 55), (w, 55), (6, 182, 212), 2)
            cv2.putText(annotated, "SignWave AI - Mots Dynamiques (LSTM Fenetre Glissante)",
                        (25, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Buffer: {len(frame_buffer)}/30 | FPS: {fps:.1f}",
                        (w - 320, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (6, 182, 212), 2)

            # Carte du mot détecté
            cv2.rectangle(annotated, (25, 80), (380, 230), (15, 23, 42), -1)
            cv2.rectangle(annotated, (25, 80), (380, 230), (6, 182, 212), 1)

            cv2.putText(annotated, "MOT DYNAMIQUE EN COURS", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
            cv2.putText(annotated, current_pred, (40, 165),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (6, 182, 212), 3)
            cv2.putText(annotated, f"Confiance LSTM : {confidence*100:.1f}%", (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("SignWave AI - Séquences Dynamiques LSTM", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()
        tts.stop()


if __name__ == "__main__":
    main()

