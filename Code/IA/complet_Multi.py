import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import json
import os
import argparse
import imageio.v3 as imageio
import torch.nn.functional as F

# Charger le fichier JSON de répartition
fichier_json = 'img_1024_seg.json'

with open(fichier_json, 'r') as f:
    repartition_data = json.load(f)

# Récupérer les ensembles train, test et validation depuis le fichier JSON
train_indices = repartition_data["train"]
test_indices = repartition_data["test"]
validation_indices = repartition_data["validation"]

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()

# Classe Dataset PyTorch
class ChangeDetectionDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        imageA_path, imageB_path, label_path = self.triplets[idx]

        # Charger les images avec imageio
        imageA = imageio.imread(imageA_path)  # Chargement en RGB par défaut
        imageB = imageio.imread(imageB_path)
        label = imageio.imread(label_path, mode="L")  # Charger en échelle de gris

        # Normaliser les valeurs entre 0 et 1
        imageA = imageA / 255.0
        imageB = imageB / 255.0
        label = label / 255.0

        # Assurer la compatibilité des types
        imageA = imageA.astype(np.float32)
        imageB = imageB.astype(np.float32)
        label = label.astype(np.float32)

        # Convertir les tableaux NumPy en tenseurs PyTorch
        imageA = torch.from_numpy(imageA).permute(2, 0, 1)  # Permuter pour obtenir (C, H, W)
        imageB = torch.from_numpy(imageB).permute(2, 0, 1)  # Permuter pour obtenir (C, H, W)
        label = torch.from_numpy(label).long()  # Convertir en long pour les cibles de classification

        # Retourner les images et le label
        return torch.cat((imageA, imageB), dim=0), label, (imageA_path, imageB_path)

# Charger le modèle UNet avec ResNet34 comme encodeur
def get_pretrained_unet():
    model = smp.Unet(
        encoder_name=args.model_name,         # Encoder pretrained on ImageNet
        encoder_weights="imagenet",
        in_channels=6,                   # 2 images avec 3 canaux chacune (RGB)
        classes=7,                       # 7 classes pour la segmentation
    )
    return model

# Fonction d'évaluation
def evaluate_model(model, data_loader, device):
    model.eval()

    # Variables pour stocker les valeurs de la matrice de confusion
    tp = np.zeros(7)  # Vrais positifs pour chaque classe
    tn = np.zeros(7)  # Vrais négatifs pour chaque classe
    fp = np.zeros(7)  # Faux positifs pour chaque classe
    fn = np.zeros(7)  # Faux négatifs pour chaque classe

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Obtenir les prédictions du modèle
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # Convertir logits en classes

            # Calcul des valeurs pour la matrice de confusion manuelle
            for i in range(7):
                tp[i] += torch.sum((predictions == i) & (labels == i)).item()
                tn[i] += torch.sum((predictions != i) & (labels != i)).item()
                fp[i] += torch.sum((predictions == i) & (labels != i)).item()
                fn[i] += torch.sum((predictions != i) & (labels == i)).item()

    # Matrice de confusion
    cm = np.array([[tn[i], fp[i], fn[i], tp[i]] for i in range(7)])

    # Calcul des métriques manuelles
    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
    iou = np.divide(tp, (tp + fp + fn), out=np.zeros_like(tp), where=(tp + fp + fn) != 0)
    accuracy = np.divide((tp + tn), (tp + tn + fp + fn), out=np.zeros_like(tp), where=(tp + tn + fp + fn) != 0)
    specificity = np.divide(tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0)

    metrics = {
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "specificity": specificity,
        "cm": cm
    }

    return metrics, cm

# Fonction pour l'entraînement et la validation
def train_and_evaluate(train_dataset, val_dataset, num_epochs=10, batch_size=2, device=None, pos_weight=None, loss_function=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_pretrained_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Choisir la fonction de pertes
    if loss_function == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(weight=pos_weight)
    elif loss_function == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_function == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_function == "DiceLoss":
        criterion = DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

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

            f1_score_epoch = np.mean(metrics["f1"])

            # Sauvegarder le meilleur modèle
            if f1_score_epoch > best_f1_score:
                best_f1_score = f1_score_epoch
                best_model = model

            # Imprimer les résultats
            print(f"Epoch {epoch+1}, F1 Score: {f1_score_epoch}")

    # Résultats de l'entraînement
    print("Training Results:")
    f1_scores = [np.mean(result['f1']) for result in results]
    print(f"Average F1 Score: {np.mean(f1_scores)} \n Max F1 Score: {np.max(f1_scores)}")

    # Ajouter des informations supplémentaires à all_results
    data = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "pos_weight": pos_weight.item() if pos_weight is not None else None,
        "loss_function": args.loss_function
    }
    all_results = {"data": data, "results": results}

    return best_model, all_results

# Main function
if __name__ == "__main__":
    # Définir les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Train and evaluate a change detection model.")
    parser.add_argument("--model_name", type=str, default="resnet34", help="Name of the model")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive weight for BCEWithLogitsLoss")
    parser.add_argument("--input_json", type=str, default="repartition_correct_512.json", help="Input JSON file with test, train, validation")
    parser.add_argument("--model_result", type=str, default="change_detection.pth", help="Save file for model extension pth")
    parser.add_argument("--metrics_result", type=str, default="all_results.npy", help="Save file for metrics extension npy")
    parser.add_argument("--loss_function", type=str, default="DiceLoss", choices=["CrossEntropyLoss", "BCEWithLogitsLoss", "MSELoss", "DiceLoss"], help="Loss function to use")

    args = parser.parse_args()

    # Charger les datasets train et validation
    train_dataset = ChangeDetectionDataset(train_indices)
    val_dataset = ChangeDetectionDataset(validation_indices)

    # Appliquer l'entraînement et la validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([args.pos_weight]).to(device) if args.pos_weight is not None else None
    best_model, all_results = train_and_evaluate(train_dataset, val_dataset, num_epochs=args.num_epochs, batch_size=2, device=device, pos_weight=pos_weight, loss_function=args.loss_function)

    # Créer le répertoire parent s'il n'existe pas
    os.makedirs(os.path.dirname(args.model_result), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_result), exist_ok=True)

    # Sauvegarder le modèle
    if best_model is not None:
        torch.save(best_model.state_dict(), args.model_result)
    else:
        print("Best model is None, not saving the model.")

    # Sauvegarder tous les résultats dans un fichier .npy
    np.save(args.metrics_result, all_results)

    # Charger et afficher les résultats sauvegardés
    loaded_results = np.load(args.metrics_result, allow_pickle=True).item()
    print(f"Model Name: {loaded_results['data']['model_name']}")
    print(f"Number of Epochs: {loaded_results['data']['num_epochs']}")
    print(f"Pos Weight: {loaded_results['data']['pos_weight']}")
    print(f"Loss Function: {loaded_results['data']['loss_function']}")
