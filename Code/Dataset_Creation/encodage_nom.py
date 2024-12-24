import os
import json

# Dossiers des images et des labels
dossier_imageA = dossier_imageB = 'DynamicEarthNet/image_sat'
dossier_label = 'DynamicEarthNet/label'

# Charger le fichier JSON de répartition
fichier_json = 'repartitions.json'

with open(fichier_json, 'r') as f:
    repartition_data = json.load(f)

# Récupérer les ensembles train, test et validation depuis le fichier JSON
train_indices = repartition_data["train"]
test_indices = repartition_data["test"]
validation_indices = repartition_data["validation"]

# Liste pour stocker les triplets pour chaque ensemble
train_triplets = []
test_triplets = []
validation_triplets = []

# Liste des zones
nom_zones = os.listdir("DynamicEarthNet/image_sat")


# Fonction pour générer les triplets
def generate_triplets():
    for nom_zone in nom_zones:
        nom_zone_reduit = nom_zone[:-3]
        for indices in train_indices:
            indiceA, indiceB = indices
            indiceA = "{:02d}".format(indiceA)
            indiceB = "{:02d}".format(indiceB)

            for suffix in range(1, 5):  # Ajouter une boucle pour les 4 zones
                # Créer les chemins des images et des labels avec les suffixes _1, _2, _3, _4
                chemin_imageA = os.path.join(dossier_imageA, nom_zone, f"{nom_zone}_{int(indiceA)*30}_rgb_{suffix}.jpeg")
                chemin_imageB = os.path.join(dossier_imageB, nom_zone, f"{nom_zone}_{int(indiceB)*30}_rgb_{suffix}.jpeg")
                chemin_label = os.path.join(dossier_label, nom_zone, f"{nom_zone_reduit}-{indiceA}_{indiceB}_{suffix}.png")
                
                # Créer le triplet et ajouter à la liste train
                train_triplets.append([chemin_imageA, chemin_imageB, chemin_label])
        
        for indices in test_indices:
            indiceA, indiceB = indices
            indiceA = "{:02d}".format(indiceA)
            indiceB = "{:02d}".format(indiceB)

            for suffix in range(1, 5):  # Ajouter une boucle pour les 4 zones
                # Créer les chemins des images et des labels avec les suffixes _1, _2, _3, _4
                chemin_imageA = os.path.join(dossier_imageA, nom_zone, f"{nom_zone}_{int(indiceA)*30}_rgb_{suffix}.jpeg")
                chemin_imageB = os.path.join(dossier_imageB, nom_zone, f"{nom_zone}_{int(indiceB)*30}_rgb_{suffix}.jpeg")
                chemin_label = os.path.join(dossier_label, nom_zone, f"{nom_zone_reduit}-{indiceA}_{indiceB}_{suffix}.png")
                
                # Créer le triplet et ajouter à la liste test
                test_triplets.append([chemin_imageA, chemin_imageB, chemin_label])
        
        for indices in validation_indices:
            indiceA, indiceB = indices
            indiceA = "{:02d}".format(indiceA)
            indiceB = "{:02d}".format(indiceB)

            for suffix in range(1, 5):  # Ajouter une boucle pour les 4 zones
                # Créer les chemins des images et des labels avec les suffixes _1, _2, _3, _4
                chemin_imageA = os.path.join(dossier_imageA, nom_zone, f"{nom_zone}_{int(indiceA)*30}_rgb_{suffix}.jpeg")
                chemin_imageB = os.path.join(dossier_imageB, nom_zone, f"{nom_zone}_{int(indiceB)*30}_rgb_{suffix}.jpeg")
                chemin_label = os.path.join(dossier_label, nom_zone, f"{nom_zone_reduit}-{indiceA}_{indiceB}_{suffix}.png")
                
                # Créer le triplet et ajouter à la liste validation
                validation_triplets.append([chemin_imageA, chemin_imageB, chemin_label])


# Fonction pour vérifier si tous les fichiers existent et compter les erreurs
def check_files_exist(triplets):
    error_count = 0  # Compteur d'erreurs
    for triplet in triplets:
        chemin_imageA, chemin_imageB, chemin_label = triplet
        if not os.path.exists(chemin_imageA):
            print(f"Fichier manquant : {chemin_imageA}")
            error_count += 1
        if not os.path.exists(chemin_imageB):
            print(f"Fichier manquant : {chemin_imageB}")
            error_count += 1
        if not os.path.exists(chemin_label):
            print(f"Fichier manquant : {chemin_label}")
            error_count += 1
    print(f" il y a {error_count}")
    return error_count


# Fonction pour enregistrer les triplets dans un fichier JSON
def save_triplets_to_json():
    # Créer un dictionnaire avec les triplets pour chaque ensemble
    triplets_data = {
        "train": train_triplets,
        "test": test_triplets,
        "validation": validation_triplets
    }

    # Enregistrer les triplets dans un fichier JSON
    with open('repartition_correct.json', 'w') as f:
        json.dump(triplets_data, f, indent=4)





# Générer les triplets
#generate_triplets()

print(len(train_indices))
print(len(test_indices))
print(len(validation_indices))


# Sauvegarder les triplets dans un fichier JSON
#save_triplets_to_json()

# Afficher un message de confirmation
# print("Les triplets ont été enregistrés dans 'repartition_correct.json'.")
