"""
Moteur d'Épellation & Assemblage de Mots
========================================
SignWave AI — Gestion du temps de maintien (Hold timer) et anti-rebond.
"""

import time


class SignSpeller:
    def __init__(self, hold_threshold: float = 1.0, min_confidence: float = 0.80):
        """
        Args:
            hold_threshold: Durée (en secondes) pendant laquelle un signe doit être maintenu
                           pour être validé et ajouté au mot en cours.
            min_confidence: Seuil de confiance minimal (entre 0.0 et 1.0).
        """
        self.hold_threshold = hold_threshold
        self.min_confidence = min_confidence

        self.current_candidate = None
        self.candidate_start_time = None

        self.current_word = ""
        self.sentence = []
        self.last_added_letter = None
        self.cooldown_until = 0.0

    def update(self, predicted_label: str, confidence: float):
        """
        Met à jour l'état avec la prédiction de la frame actuelle.
        
        Returns:
            dict: {
                "letter_added": str or None,
                "progress": float (0.0 à 1.0),
                "current_word": str,
                "full_text": str,
                "candidate": str or None
            }
        """
        now = time.time()
        result = {
            "letter_added": None,
            "progress": 0.0,
            "current_word": self.current_word,
            "full_text": self.get_full_text(),
            "candidate": self.current_candidate
        }

        # Période de repos après validation d'une lettre
        if now < self.cooldown_until:
            return result

        if confidence < self.min_confidence or predicted_label is None:
            self.current_candidate = None
            self.candidate_start_time = None
            return result

        # Si même signe que le candidat en cours
        if predicted_label == self.current_candidate:
            elapsed = now - self.candidate_start_time
            progress = min(1.0, elapsed / self.hold_threshold)
            result["progress"] = progress

            if elapsed >= self.hold_threshold:
                # Validation du caractère
                added = self._commit_char(predicted_label)
                result["letter_added"] = added
                result["current_word"] = self.current_word
                result["full_text"] = self.get_full_text()
                result["progress"] = 0.0

                # Réinitialisation et cooldown court (0.5s)
                self.current_candidate = None
                self.candidate_start_time = None
                self.cooldown_until = now + 0.5
        else:
            # Nouveau candidat
            self.current_candidate = predicted_label
            self.candidate_start_time = now
            result["candidate"] = predicted_label
            result["progress"] = 0.0

        return result

    def _commit_char(self, char: str) -> str:
        char_upper = str(char).upper()
        if char_upper in ["SPACE", "ESPACE", "_"]:
            if self.current_word:
                self.sentence.append(self.current_word)
                self.current_word = ""
            return " "
        elif char_upper in ["DEL", "DELETE", "BACKSPACE"]:
            if self.current_word:
                self.current_word = self.current_word[:-1]
            elif self.sentence:
                self.current_word = self.sentence.pop()
            return "<DEL>"
        else:
            self.current_word += char_upper
            return char_upper

    def commit_word(self):
        """Valide manuellement le mot en cours dans la phrase."""
        if self.current_word:
            self.sentence.append(self.current_word)
            word = self.current_word
            self.current_word = ""
            return word
        return None

    def clear(self):
        self.current_word = ""
        self.sentence = []
        self.current_candidate = None
        self.candidate_start_time = None

    def get_full_text(self) -> str:
        words = list(self.sentence)
        if self.current_word:
            words.append(self.current_word)
        return " ".join(words)
