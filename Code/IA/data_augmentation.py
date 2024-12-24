import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import json
import os
from torchvision import transforms

# Charger le fichier JSON de répartition
fichier_json = 'img_1024.json'

with open(fichier_json, 'r') as f:
    repartition_data = json.load(f)

# Récupérer les ensembles train, test et validation depuis le fichier JSON
train_indices = repartition_data["train"]
validation_indices = repartition_data["validation"]

class ChangeDetectionDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        imageA_path, imageB_path, label_path = self.triplets[idx]

        # Charger les images
        imageA = Image.open(imageA_path).convert('RGB')
        imageB = Image.open(imageB_path).convert('RGB')

        # Charger le label (image binaire)
        label = Image.open(label_path).convert('L')

        # Appliquer les transformations
        if self.transform:
            seed = torch.random.seed()
            imageA = self.apply_transform(imageA, seed)
            imageB = self.apply_transform(imageB, seed)
            label = self.apply_transform(label, seed)

        # Retourner les images et le label
        return torch.cat((imageA, imageB), dim=0), label, (imageA_path, imageB_path)

    def apply_transform(self, image, seed):
        torch.random.manual_seed(seed)
        return self.transform(image)

    def get_label_name(self, idx):
        # Extrait le nom du label à partir du chemin du fichier
        label_path = self.triplets[idx][2]
        return label_path.split('/')[-1]  # Exemple : extrait le nom du fichier

    def get_zone(self, idx):
        # Extrait la zone à partir des 9 premiers caractères du nom du label
        label_name = self.get_label_name(idx)
        return label_name[:9]

# Définir les transformations
transform = transforms.Compose([
    transforms.RandomCrop((512, 512)),  # Crop aléatoire de 512x512
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
    transforms.RandomVerticalFlip(),    # Retournement vertical aléatoire
    transforms.RandomRotation(10),      # Rotation aléatoire de ±10 degrés
    transforms.ToTensor()               # Convertir en tenseur
])
# Charger le modèle UNet avec ResNet34 comme encodeur
def get_pretrained_unet():
    model = smp.Unet(
        encoder_name="resnet34",         # Encoder pretrained on ImageNet
        encoder_weights="imagenet",
        in_channels=6,                   # 2 images avec 3 canaux chacune (RGB)
        classes=1,                       # 1 sortie pour la carte de détection de changement
    )
    return model






# Fonction d'évaluation
def evaluate_model(model, data_loader, device):
    model.eval()

    # Variables pour stocker les valeurs de la matrice de confusion
    tp = 0  # Vrais positifs
    tn = 0  # Vrais négatifs
    fp = 0  # Faux positifs
    fn = 0  # Faux négatifs


    with torch.no_grad():
        for images, labels, _ in data_loader:
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


    # Matrice de confusion
    cm = np.array([[tn, fp], [fn, tp]])

    # Calcul des métriques manuelles
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0



    print(f"Manual F1-score: {f1}")
    print(f"IoU: {iou}")
    print(f"Accuracy: {accuracy}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")

    metrics = {
        "f1": f1,   
        "iou": iou,
        "accuracy": accuracy,
        "specificity": specificity,
        "cm": cm
    }

    return metrics, cm



# Fonction pour l'entraînement et la validation
def train_and_evaluate(train_dataset, val_dataset, num_epochs=10, batch_size=2, device=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_pretrained_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy pour la détection de changement

    best_model = None   
    best_f1_score = 0
    results = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calcul de la perte
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Afficher la perte pour chaque époque
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Évaluation toutes les 3 époques
        if (epoch + 1) % 3 == 0:
            print("Evaluating...")
            metrics, cm = evaluate_model(model, val_loader, device)
            metrics['epoch'] = epoch + 1

            results.append(metrics)

            f1_score_epoch = metrics["f1"]

            # Sauvegarder le meilleur modèle
            if f1_score_epoch > best_f1_score:
                best_f1_score = f1_score_epoch
                best_model = model

            # Imprimer les résultats
            print(f"Epoch {epoch+1}, F1 Score: {f1_score_epoch}, Confusion Matrix:\n{cm}")

    # Résultats de l'entraînement
    print("Training Results:")
    f1_scores = [result['f1'] for result in results]
    print(f"Average F1 Score: {np.mean(f1_scores)}")
    print(f"Standard Deviation: {np.std(f1_scores)}")

    return best_model, results

# Main function
if __name__ == "__main__":
    # Charger les datasets train et validation
    train_dataset = ChangeDetectionDataset(train_indices, transform=transform)
    val_dataset = ChangeDetectionDataset(validation_indices, transform=transform)

    # Appliquer l'entraînement et la validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 24
    best_model, all_results = train_and_evaluate(train_dataset, val_dataset, num_epochs=num_epochs, batch_size=2, device=device)

    # Sauvegarder le modèle 
    torch.save(best_model.state_dict(), 'change_detection.pth')

    # Sauvegarder tous les résultats dans un fichier .npy
    np.save('all_results.npy', all_results)

    # Charger et afficher les résultats sauvegardés
    loaded_results = np.load('all_results.npy', allow_pickle=True)
    for result in loaded_results:
        print(f"Epoch: {result['epoch']}, F1 Score: {result['f1']}, Confusion Matrix:{result['cm']}")
