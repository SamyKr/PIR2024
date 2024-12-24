import numpy as np
from PIL import Image
import os
from itertools import combinations




def carte_chang_binaire(nom_zone):
    # Production des cartes de changements binaires par couples de mois, créés par combinaison
    indices = list(range(24))  # Générer une liste d'indices de 0 à 23

    # Créer les paires d'indices (combinaisons)
    pairs = list(combinations(indices, 2))

    # Formater chaque élément des paires avec deux chiffres
    pairs = [(f"{i:02}", f"{j:02}") for i, j in pairs]

    # Charger les labels de la zone
    mx_label = np.load(f'Code/DynamicEarthNet/labels/{nom_zone}_13.npy')

    # Créer un répertoire pour la zone si il n'existe pas déjà
    output_dir = f"DynamicEarthNet/label/{nom_zone}_13"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Boucle à partir de l'indice 20 dans la liste des paires
    for x in range(0, 21):
        # Accéder aux indices formatés pour chaque paire de mois
        indiceA, indiceB = pairs[x]

        # Comparer les valeurs des mois pour générer la carte de changement binaire
        array_change = (mx_label[int(indiceA)][:][:] == mx_label[int(indiceB)][:][:])

        # Créer l'image de changement binaire
        image_change = Image.fromarray(array_change.astype(np.uint8))

        # Inverser les couleurs (si 0 devient 255, et inversement)
        image_change = image_change.point(lambda x: 255 if x == 0 else 0)

        # Construire le chemin de sauvegarde du fichier
        image_path = os.path.join(output_dir, f"{nom_zone}-{indiceA}_{indiceB}.png")

        # Sauvegarder l'image générée
        image_change.save(image_path)
    

# Liste des zones (assurez-vous que les noms des dossiers sont corrects)
dossiers_zones = os.listdir("DynamicEarthNet/image_sat")

# Générer les cartes de changement pour chaque zone


for zone in dossiers_zones:
    # Appeler la fonction avec le nom de zone, en prenant seulement les 9 premiers caractères du nom du dossier
    carte_chang_binaire(zone[:9])


