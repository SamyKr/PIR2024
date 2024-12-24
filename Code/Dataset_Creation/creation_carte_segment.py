import numpy as np
from PIL import Image
import os
from itertools import combinations

def carte_chang_semantique(nom_zone):
    # Liste des couleurs à associer à chaque valeur d'indiceB (7 valeurs différentes)
    couleurs = {
        0: (0, 0, 0),          # Noir pour indiceB == 1
        1: (255, 255, 0),      # Jaune pour indiceB == 2
        2: (0, 255, 0),        # Vert pour indiceB == 3
        3: (0, 0, 139),        # Bleu foncé pour indiceB == 4
        4: (139, 69, 19),      # Marron pour indiceB == 5
        5: (135, 206, 250),    # Bleu ciel pour indiceB == 6
        6: (173, 216, 230)     # Bleu glace pour indiceB == 7
    }

    # Production des cartes de changements par couples de mois, créés par combinaison
    indices = list(range(24))  # Générer une liste d'indices de 0 à 23

    # Créer les paires d'indices (combinaisons)
    pairs = list(combinations(indices, 2))

    # Formater chaque élément des paires avec deux chiffres
    pairs = [(f"{i:02}", f"{j:02}") for i, j in pairs]

    # Charger les labels de la zone
    mx_label = np.load(f'checkpoints/labels/{nom_zone}_13.npy')

    # Créer un dossier pour sauvegarder les images
    output_dir = f"img1024_seg/labels_seg/{nom_zone}"
    os.makedirs(output_dir, exist_ok=True)

    for indiceA, indiceB in pairs:
        # Construire le chemin complet pour enregistrer l'image
        image_path = os.path.join(output_dir, f"{nom_zone}_{indiceA}_{indiceB}.png")

        # Vérifier si l'image existe déjà
        if os.path.exists(image_path):
            print(f"Image déjà existante : {image_path}")
            continue

        # Récupérer les labels pour les indices A et B
        label_A = mx_label[int(indiceA)]
        label_B = mx_label[int(indiceB)]

        # Créer une image vide pour la carte de changement sémantique
        image_change = np.zeros((label_A.shape[0], label_A.shape[1], 3), dtype=np.uint8)

        # Parcourir chaque pixel et déterminer la couleur à utiliser
        for i in range(label_A.shape[0]):
            for j in range(label_A.shape[1]):
                # Si les labels sont différents, associer la couleur basée sur indiceB
                if label_A[i, j] != label_B[i, j]:
                    couleur = couleurs.get(label_B[i, j], (0, 0, 0))  # Noir si l'indiceB n'est pas trouvé
                    image_change[i, j] = couleur

        # Créer l'image à partir du tableau numpy
        image_change_pil = Image.fromarray(image_change)

        # Sauvegarder l'image
        image_change_pil.save(image_path)

        print(f"Image enregistrée : {image_path}")

# Liste des zones (assurez-vous que les noms des dossiers sont corrects)
dossiers_zones = os.listdir("DynamicEarthNet/image_sat")

# Générer et enregistrer les cartes de changement pour chaque zone
for zone in dossiers_zones:
    # Appeler la fonction avec le nom de zone, en prenant seulement les 9 premiers caractères du nom du dossier
    carte_chang_semantique(zone[:9])
