"""
Étape 2 : Entraînement & Évaluation Comparative des Modèles
===========================================================
SignWave AI — Classification Statique (Alphabet A-Z & Chiffres 0-9)

Entraîne et compare :
  1. Random Forest Classifier
  2. Support Vector Machine (SVM RBF)
  3. Multi-Layer Perceptron (MLP Dense)

Génère la matrice de confusion, les métriques d'évaluation et sauvegarde le meilleur modèle.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode headless pour génération d'images sans écran
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    print("=" * 65)
    print("  SignWave AI — Étape 2 : Entraînement & Benchmark Scientifique")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[ERREUR] Données introuvables. Exécutez d'abord 'step1_extract_landmarks.py'.")
        return

    # Chargement des données
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"\n📦 Données chargées : {X.shape[0]} exemples, {X.shape[1]} caractéristiques par main.")

    # Encodage des classes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    print(f"🎯 Nombre de classes : {len(classes)} ({list(classes[:8])}...)")

    # Découpage Train / Test (80% / 20% stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"📊 Répartition : {len(X_train)} train | {len(X_test)} test\n")

    # Définition des modèles à comparer (idéal pour la soutenance 2 IRM)
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, n_jobs=-1),
        "SVM (Kernel RBF)": SVC(C=10.0, kernel='rbf', probability=True, random_state=42),
        "MLP (Dense Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    }

    results = {}
    best_acc = 0.0
    best_name = None
    best_model = None

    print(f"{'Modèle':<25} | {'Train Acc':<10} | {'Test Acc':<10} | {'Latence Inférence':<16}")
    print("-" * 70)

    for name, model in candidates.items():
        # Entraînement
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Évaluation
        train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
        
        # Test de latence unitaire
        t_start = time.perf_counter()
        y_pred = model.predict(X_test)
        t_end = time.perf_counter()
        latency_ms = ((t_end - t_start) / len(X_test)) * 1000.0

        test_acc = accuracy_score(y_test, y_pred) * 100

        results[name] = {
            "train_accuracy": round(train_acc, 2),
            "test_accuracy": round(test_acc, 2),
            "latency_ms": round(latency_ms, 3),
            "train_time_sec": round(train_time, 2)
        }

        print(f"{name:<25} | {train_acc:>8.2f}% | {test_acc:>8.2f}% | {latency_ms:>12.3f} ms/échantillon")

        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name
            best_model = model

    print("-" * 70)
    print(f"🏆 Meilleur modèle retenu : {best_name} ({best_acc:.2f}% de précision)")

    # Rapport détaillé sur le meilleur modèle
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best, target_names=classes, output_dict=True)

    # Sauvegarde du modèle et du label encoder
    model_save_path = os.path.join(models_dir, "static_classifier.pkl")
    encoder_save_path = os.path.join(models_dir, "label_encoder.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(encoder_save_path, "wb") as f:
        pickle.dump(label_encoder, f)

    # Génération et sauvegarde de la Matrice de Confusion
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=classes, yticklabels=classes, cbar=True
    )
    plt.title(f"SignWave AI — Matrice de Confusion ({best_name} - Précision: {best_acc:.1f}%)", fontsize=14, pad=15)
    plt.xlabel("Classe Prédite", fontsize=12)
    plt.ylabel("Classe Réelle", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Sauvegarde des métriques JSON
    summary = {
        "best_model_name": best_name,
        "best_accuracy": best_acc,
        "num_classes": len(classes),
        "classes": list(classes),
        "benchmark_comparison": results,
        "classification_report": report
    }
    summary_path = os.path.join(models_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n📁 Fichiers sauvegardés :")
    print(f"   • Modèle sérialisé : {model_save_path}")
    print(f"   • Encodeur de classes : {encoder_save_path}")
    print(f"   • Graphique matrice de confusion : {cm_path}")
    print(f"   • Résumé JSON pour le jury : {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
