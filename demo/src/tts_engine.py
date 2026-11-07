"""
Moteur de Synthèse Vocale (Text-To-Speech)
=========================================
SignWave AI — Restitution audio asynchrone non bloquante.
"""

import threading
import queue
import time


class TTSEngine:
    def __init__(self, rate: int = 160, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.queue = queue.Queue()
        self.is_running = True
        self._last_spoken = ""
        self._last_spoken_time = 0.0

        # Démarre le thread de lecture audio
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception as e:
            print(f"[TTS] Avertissement: pyttsx3 n'a pas pu être initialisé: {e}")

        while self.is_running:
            try:
                text = self.queue.get(timeout=0.2)
                if text is None:
                    break

                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        print(f"[TTS] Erreur de lecture: {err}")
                else:
                    print(f"[TTS Audio Simulée] : {text}")

                self.queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text: str, min_interval_seconds: float = 1.5):
        """
        Enfile une parole si le texte est différent du précédent ou après l'intervalle minimal.
        """
        text = text.strip()
        if not text:
            return

        now = time.time()
        if text != self._last_spoken or (now - self._last_spoken_time) > min_interval_seconds:
            self._last_spoken = text
            self._last_spoken_time = now
            self.queue.put(text)

    def stop(self):
        self.is_running = False
