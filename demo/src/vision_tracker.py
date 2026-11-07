"""
Module de Détection et Visualisation des Repères 3D
===================================================
SignWave AI — MediaPipe Hands Tracker optimisé.
"""

import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame_bgr, draw_styled: bool = True):
        """
        Traite une image BGR de la webcam et retourne les repères et l'image annotée.
        
        Returns:
            multi_hand_landmarks: Liste des mains détectées (ou None)
            annotated_frame: Image BGR avec tracé stylisé
        """
        # Calcul du FPS
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = current_time

        # Conversion RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        annotated_frame = frame_bgr.copy()

        if results.multi_hand_landmarks and draw_styled:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tracé des connexions osseuses avec style cyberpunk/technologique
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(6, 182, 212), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(241, 102, 99), thickness=2, circle_radius=2)
                )

        return results.multi_hand_landmarks, annotated_frame

    def get_fps(self) -> float:
        return self.fps

    def release(self):
        self.hands.close()
