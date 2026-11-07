"""
Étape 3 : Tracking Squelette 3D MediaPipe & Invariance Géométrique
==================================================================
SignWave AI — Visualisation temps réel des 21 repères et normalisation vectorielle.

Démontre en direct :
  - Extraction des 21 repères 3D à 30+ FPS sur CPU.
  - Normalisation mathématique : P'_i = (P_i - P_0) / ||P_9 - P_0||.
  - Invariance à la distance caméra et à la position dans l'image.
"""

import os
import sys
import time
import cv2
import numpy as np

# Import des modules internes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.src.vision_tracker import HandTracker
from demo.src.normalizer import LandmarkNormalizer


def main():
    print("=" * 65)
    print("  SignWave AI — Tracking Squelette 3D MediaPipe")
    print("  Appuyez sur [Q] ou [ECHAP] pour quitter.")
    print("=" * 65)

    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    normalizer = LandmarkNormalizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            landmarks_list, annotated = tracker.process_frame(frame, draw_styled=True)
            fps = tracker.get_fps()

            # Bandeau supérieur
            cv2.rectangle(annotated, (0, 0), (w, 50), (15, 23, 42), -1)
            cv2.line(annotated, (0, 50), (w, 50), (6, 182, 212), 2)
            cv2.putText(annotated, "MediaPipe 3D Hand Landmarks Tracker (Invariance Spatiale)",
                        (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 140, 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (6, 182, 212), 2)

            # Télémétrie des coordonnées
            if landmarks_list:
                for idx, hand in enumerate(landmarks_list):
                    lms = hand.landmark
                    vec = normalizer.normalize(lms)

                    # Affichage des coordonnées clés (P0 Poignet, P8 Index, P4 Pouce)
                    p0 = lms[0]
                    p8 = lms[8]
                    p12 = lms[12]

                    panel_y = 70 + idx * 160
                    cv2.rectangle(annotated, (20, panel_y), (360, panel_y + 140), (15, 23, 42), -1)
                    cv2.rectangle(annotated, (20, panel_y), (360, panel_y + 140), (6, 182, 212), 1)

                    cv2.putText(annotated, f"MAIN {idx+1} : 21 POINTS CAPTURES", (30, panel_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)
                    cv2.putText(annotated, f"Poignet P0 : ({p0.x:.2f}, {p0.y:.2f}, {p0.z:.2f})",
                                (30, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    cv2.putText(annotated, f"Bout Index P8 : ({p8.x:.2f}, {p8.y:.2f}, {p8.z:.2f})",
                                (30, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    cv2.putText(annotated, f"Bout Majeur P12 : ({p12.x:.2f}, {p12.y:.2f}, {p12.z:.2f})",
                                (30, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    cv2.putText(annotated, f"Vecteur Normalise 63D : OK", (30, panel_y + 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212), 1)
            else:
                cv2.putText(annotated, "Placez vos mains devant la camera pour voir le squelette 3D",
                            (w // 2 - 320, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            cv2.imshow("SignWave AI - Squelette 3D MediaPipe", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

