"""
Normalisation Géométrique Vectorielle
====================================
SignWave AI — Module de prétraitement pour l'invariance spatiale.

Formule appliquée :
    P'_i = (P_i - P_0) / ||P_9 - P_0||

- P_0 : Poignet (Wrist)
- P_9 : Base du majeur (Middle Finger MCP)
- P_i : Chaque point parmi les 21 repères 3D MediaPipe
"""

import numpy as np


class LandmarkNormalizer:
    def __init__(self):
        self.wrist_idx = 0
        self.ref_joint_idx = 9  # Base du majeur

    def normalize(self, landmarks):
        """
        Normalise un tableau de repères 3D de taille (21, 3).
        
        Args:
            landmarks: array-like de forme (21, 3) ou liste d'objets MediaPipe avec attributs x, y, z.
            
        Returns:
            np.ndarray de forme (63,) aplati, centré sur le poignet et mis à l'échelle.
        """
        # Conversion en tableau NumPy (21, 3)
        if not isinstance(landmarks, np.ndarray):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        else:
            coords = np.array(landmarks, dtype=np.float32)

        if coords.shape != (21, 3):
            raise ValueError(f"Les repères doivent avoir la forme (21, 3), reçu: {coords.shape}")

        # 1. Centrage au poignet (Point 0 devient l'origine 0,0,0)
        wrist = coords[self.wrist_idx].copy()
        centered = coords - wrist

        # 2. Facteur d'échelle : Distance Euclidienne entre le poignet (0) et le majeur (9)
        ref_distance = np.linalg.norm(centered[self.ref_joint_idx])
        if ref_distance < 1e-6:
            # Sécurité en cas de points dégénérés : division par la distance maximale
            ref_distance = np.max(np.linalg.norm(centered, axis=1))
            if ref_distance < 1e-6:
                ref_distance = 1.0

        normalized = centered / ref_distance

        # 3. Aplatissement en vecteur 1D de 63 valeurs
        return normalized.flatten()
