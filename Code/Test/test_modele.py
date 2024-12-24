import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
from entrainement_test_validation import ChangeDetectionDataset, get_pretrained_unet, evaluate_model  # Import des fonctions et classes du fichier train.py

# Charger le fichier JSON de répartition
fichier_json = 'repartition_correct_512.json'

with open(fichier_json, 'r') as f:
    repartition_data = json.load(f)

# Récupérer les indices du jeu de test
test_indices = repartition_data["test"]

# Définir les transformations pour le test
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Charger le jeu de test
test_dataset = ChangeDetectionDataset(test_indices, transform=transform)

# Fonction de chargement du modèle
def load_model(model_path, device):
    model = get_pretrained_unet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Fonction de test sur le jeu de test
def test_model(test_dataset, model, device):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialisation des compteurs pour la matrice de confusion
    tp = 0  # Vrais positifs
    tn = 0  # Vrais négatifs
    fp = 0  # Faux positifs
    fn = 0  # Faux négatifs

    f_scores = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Obtenir les prédictions du modèle
            outputs = model(images)
            predictions = torch.sigmoid(outputs)  # Convertir logits en probabilités

            # Binariser les prédictions
            predictions_binary = (predictions >= 0.5).float()

            # Binariser également les labels
            labels_binary = (labels >= 0.5).float()

            # Calcul des valeurs pour la matrice de confusion manuelle
            tp += torch.sum((predictions_binary == 1) & (labels_binary == 1)).item()
            tn += torch.sum((predictions_binary == 0) & (labels_binary == 0)).item()
            fp += torch.sum((predictions_binary == 1) & (labels_binary == 0)).item()
            fn += torch.sum((predictions_binary == 0) & (labels_binary == 1)).item()

            # Calcul du F-score pour cet échantillon
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            f_scores.append(f1)

    # Matrice de confusion
    cm = np.array([[tn, fp], [fn, tp]])

    # Calcul des métriques manuelles
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    metrics = {'f1': f1, 'precision': precision, 'recall': recall}

    return metrics, cm, f_scores

# Charger le modèle et tester
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'resultats/resnet101_18epochs/resnet101_18epochs.pth'
model = load_model(model_path, device)

# Tester le modèle sur le jeu de test
metrics, cm, f_scores = test_model(test_dataset, model, device)

# Afficher les résultats du test
print("\nRésultats du test:")
print(f"F1 Score (manuel): {metrics['f1']:.4f}")
print(f"Precision (manuel): {metrics['precision']:.4f}")
print(f"Recall (manuel): {metrics['recall']:.4f}")
print("Matrice de Confusion (manuel):")
print(cm)

# Calculer la moyenne des F1-scores
mean_f1_score = np.mean(f_scores)

# Tracer la répartition des F-scores
plt.figure(figsize=(10, 6))
plt.hist(f_scores, bins=20, edgecolor='black')
plt.title('Répartition des F-scores')
plt.xlabel('F-score')
plt.ylabel('Fréquence')
plt.axvline(mean_f1_score, color='r', linestyle='dashed', linewidth=1)
plt.text(mean_f1_score + 0.01, plt.ylim()[1] * 0.9, f'Moyenne: {mean_f1_score:.4f}', color='r')
plt.grid(True)
plt.show()

# Fonction pour afficher les résultats
def plot_predictions(model, test_loader, device):
    model.eval()
    results_by_zone = {}

    with torch.no_grad():
        for i, (inputs, labels, *_) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Binariser les prédictions
            predictions = (outputs.sigmoid() > 0.5).cpu().numpy().squeeze()
            labels = labels.cpu().numpy().squeeze()

            # Calculer le score F1
            f1 = f1_score(labels.flatten(), predictions.flatten(), average='binary')
            zone = test_dataset.get_zone(i)  # Obtenir la zone

            if zone not in results_by_zone:
                results_by_zone[zone] = []

            results_by_zone[zone].append((f1, inputs.cpu(), labels, predictions, i))

    # Sélectionner le meilleur résultat pour chaque zone
    best_results = {zone: max(results, key=lambda x: x[0]) for zone, results in results_by_zone.items()}

    # Afficher les meilleurs résultats par zone
    for zone, (f1, inputs, labels, predictions, idx) in best_results.items():
        label_name = test_dataset.get_label_name(idx)  # Obtenir le nom du label

        plt.figure(figsize=(8, 4))

        # Label réel
        plt.subplot(1, 2, 1)
        plt.imshow(labels, cmap='gray')
        plt.title(f"Label réel: {label_name}\nZone: {zone}")  # Ajouter le nom du label et la zone
        plt.axis("off")

        # Prédiction
        plt.subplot(1, 2, 2)
        plt.imshow(predictions, cmap='gray')
        plt.title(f"Prédiction IA (F1: {f1:.4f})")
        plt.axis("off")

        plt.show()

# Chargement des données de test
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Afficher le meilleur résultat par zone
#plot_predictions(model, test_loader, device)

