from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
from PIL import Image
import numpy as np
import json
import os
from torchvision import transforms

# Charger le fichier JSON de répartition
fichier_json = 'repartition.json'

with open(fichier_json, 'r') as f:
    repartition_data = json.load(f)

# Récupérer les ensembles train, test et validation depuis le fichier JSON
train_indices = repartition_data["train"]
test_indices = repartition_data["test"]
validation_indices = repartition_data["validation"]

# Classe Dataset PyTorch
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
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
            label = self.transform(label)

        # Retourner les images et le label
        return torch.cat((imageA, imageB), dim=0), label, (imageA_path, imageB_path)

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Charger le modèle UNet avec ResNet34 comme encodeur
def get_pretrained_unet():
    model = smp.Unet(
        encoder_name="resnet18",         # Encoder pretrained on ImageNet
        encoder_weights="imagenet",
        in_channels=6,                   # 2 images avec 3 canaux chacune (RGB)
        classes=1,                       # 1 sortie pour la carte de détection de changement
    )
    return model

# Fonction d'évaluation
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Obtenir les prédictions du modèle
            outputs = model(images)
            predictions = torch.sigmoid(outputs)  # Convertir logits en probabilités

            # Binariser les prédictions
            predictions_binary = (predictions >= 0.5).float()  # Binariser les prédictions
            labels = labels.cpu().numpy().flatten()

            # Binariser également les labels (en cas de valeurs continues)
            labels_binary = (labels >= 0.5).astype(int)

            # Ajouter les résultats aux listes
            y_true.extend(labels_binary)
            y_pred.extend(predictions_binary.cpu().numpy().flatten())

    # Calcul des métriques
    f1 = f1_score(y_true, y_pred)  # Calcul du F1 Score
    cm = confusion_matrix(y_true, y_pred)  # Calcul de la matrice de confusion
    return f1, cm
    #return f1

# Fonction pour l'entraînement et la validation croisée
def train_and_evaluate_kfold(dataset, k=5, num_epochs=10, batch_size=2, device=None):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    best_model = None
    best_f1_score = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")

        # Créer les ensembles d'entraînement et de validation pour ce pli
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Créer les DataLoaders pour ce pli
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        model = get_pretrained_unet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy pour la détection de changement

        # Entraînement pour ce pli
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

            # Évaluation toutes les 2 époques
            if (epoch + 1) % 2 == 0:
                print("Evaluating...")
                f1_score_fold, cm = evaluate_model(model, val_loader, device)
                fold_results.append({
                    'epoch': epoch + 1,
                    'fold': fold + 1,
                    'f1_score': f1_score_fold,
                    'confusion_matrix': cm
                })

                # Sauvegarder le meilleur modèle
                if f1_score_fold > best_f1_score:
                    best_f1_score = f1_score_fold
                    best_model = model

                # Imprimer les résultats    
                print(f"Fold {fold+1}, Epoch {epoch+1}, F1 Score: {f1_score_fold}, Confusion Matrix:\n{cm}")
                #print(f"Fold {fold+1}, Epoch {epoch+1}, F1 Score: {f1_score_fold}")

    # Résultats de la validation croisée
    print("K-Fold Cross Validation Results:")
    f1_scores = [result['f1_score'] for result in fold_results]
    print(f"Average F1 Score: {np.mean(f1_scores)}")
    print(f"Standard Deviation: {np.std(f1_scores)}")

    return best_model, fold_results

# Charger le dataset complet (train + validation)
full_dataset = ChangeDetectionDataset(train_indices + validation_indices, transform=transform)

# Appliquer la validation croisée
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 13
best_model, all_results = train_and_evaluate_kfold(full_dataset, k=5, num_epochs=num_epochs, batch_size=2, device=device)

# Sauvegarder le modèle
torch.save(best_model.state_dict(), 'change_detection.pth')

# Sauvegarder tous les résultats dans un fichier .npy
np.save('all_results.npy', all_results)

# Charger et afficher les résultats sauvegardés
loaded_results = np.load('all_results.npy', allow_pickle=True)
for result in loaded_results:
    print(f"Fold: {result['fold']}, Epoch: {result['epoch']}, F1 Score: {result['f1_score']}, Confusion Matrix:\n{result['confusion_matrix']}")
