"""
Étape 1 : Entraînement Deep Learning sur le DATASET (36 Classes)
================================================================
SignWave AI — Classification d'images par Réseau Convolutif (PyTorch GPU/CPU)

Entraîne un réseau convolutif optimisé (SignNet) sur les 36 classes (0-9 et A-Z)
du dossier DATASET/ avec accélération CUDA.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Architecture CNN légère et ultra-rapide (< 10 ms par image)
class SignNet(nn.Module):
    def __init__(self, num_classes=36):
        super(SignNet, self).__init__()
        self.features = nn.Sequential(
            # Bloc 1 : 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloc 2 : 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloc 3 : 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloc 4 : 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 65)
    print("  SignWave AI — Entraînement Deep Learning sur DATASET/")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Périphérique de calcul : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "..", "DATASET")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(base_dir, "DATASET")

    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Transformations avec Data Augmentation pour éviter l'overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Chargement du dataset complet
    full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"📊 Dataset chargé : {len(full_dataset)} images réparties sur {num_classes} classes.")

    # Sauvegarde de la liste des classes
    classes_path = os.path.join(models_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    # Découpage 80% train / 20% validation
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.80 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Remplacement de la transformation sur le set de validation
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Modèle, loss, optimizer
    model = SignNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    epochs = 20
    best_acc = 0.0
    model_save_path = os.path.join(models_dir, "sign_classifier.pth")

    print("\nDébut de l'entraînement...\n")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # Évaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        val_acc = (correct_val / total_val) * 100
        val_loss = val_loss / total_val

        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    elapsed = time.time() - t0
    print(f"\n✅ Entraînement terminé en {elapsed:.1f} secondes !")
    print(f"🏆 Meilleure précision de validation : {best_acc:.2f}%")

    # Évaluation finale & Matrice de Confusion
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"SignWave AI — Matrice de Confusion CNN ({best_acc:.1f}%)", fontsize=14)
    plt.xlabel("Prédit", fontsize=12)
    plt.ylabel("Réel", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"📁 Modèle sauvegardé : {model_save_path}")
    print(f"📁 Matrice de confusion : {cm_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

