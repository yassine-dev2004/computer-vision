"""
Étape 1 : Extraction & Normalisation des Repères 3D
====================================================
SignWave AI — Traitement par lot du dossier DATASET/

Parcourt les sous-dossiers (0-9, A-Z), extrait les 21 points 3D de chaque main via MediaPipe,
applique la normalisation vectorielle, et sauvegarde le dataset vectoriel résultant.
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd

# Ajout du dossier parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.src.normalizer import LandmarkNormalizer


def main():
    print("=" * 65)
    print("  SignWave AI — Étape 1 : Extraction des Repères 3D MediaPipe")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "..", "DATASET")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(base_dir, "DATASET")

    if not os.path.exists(dataset_dir):
        print(f"[ERREUR] Le dossier DATASET n'a pas été trouvé à : {dataset_dir}")
        return

    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Récupération des sous-dossiers de classes (0-9, A-Z)
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"\n📂 Dossier source : {dataset_dir}")
    print(f"📊 Classes trouvées ({len(classes)}) : {', '.join(classes[:10])} ... {', '.join(classes[-5:])}")

    # Initialisation de MediaPipe Hands
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    normalizer = LandmarkNormalizer()

    X_data = []
    y_data = []
    metadata = []

    total_images = 0
    detected_count = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(dataset_dir, class_name)
            image_paths = glob.glob(os.path.join(class_folder, "*.*"))
            valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_exts]

            class_detected = 0
            for img_path in image_paths:
                total_images += 1
                img = cv2.imread(img_path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    features = normalizer.normalize(landmarks)

                    X_data.append(features)
                    y_data.append(class_name)
                    metadata.append({
                        "file": os.path.basename(img_path),
                        "class": class_name
                    })

                    detected_count += 1
                    class_detected += 1

            # Progression par classe
            pct = (class_detected / len(image_paths) * 100) if image_paths else 0.0
            print(f"  [{class_idx+1:02d}/{len(classes):02d}] Classe '{class_name}' : {class_detected}/{len(image_paths)} mains détectées ({pct:.1f}%)")

    if not X_data:
        print("\n[ERREUR] Aucune main n'a pu être détectée.")
        return

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data)

    print("\n" + "=" * 65)
    print(f"✅ Extraction terminée !")
    print(f"   • Images analysées  : {total_images}")
    print(f"   • Mains extraites   : {detected_count} ({detected_count/total_images*100:.1f}%)")
    print(f"   • Matrice X (Features) : {X.shape} (63 coordonnées par main)")
    print(f"   • Matrice y (Labels)   : {y.shape}")

    # Sauvegarde NumPy
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    # Sauvegarde CSV tabulaire
    columns = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    df = pd.DataFrame(X, columns=columns)
    df["label"] = y
    csv_path = os.path.join(output_dir, "landmarks_data.csv")
    df.to_csv(csv_path, index=False)

    # Sauvegarde des classes ordonnées
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"📁 Fichiers générés dans '{output_dir}' :")
    print(f"   - X.npy & y.npy (matrices rapides pour entraînement)")
    print(f"   - landmarks_data.csv ({os.path.getsize(csv_path)/1024:.1f} Ko)")
    print(f"   - classes.json ({len(classes)} classes)")
    print("=" * 65)


if __name__ == "__main__":
    main()
