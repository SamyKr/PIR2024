import json
from PIL import Image
import os

def decouper_image(image_path):
    # Ouvrir l'image
    img = Image.open(image_path)

    # Vérifier que l'image est de taille 1024x1024
    if img.size != (1024, 1024):
        print(f"L'image {image_path} n'a pas la taille correcte (1024x1024).")
        return None

    # Découper l'image en 4 parties (chaque morceau de 512x512)
    width, height = img.size
    images_decoupees = [
        img.crop((0, 0, width // 2, height // 2)),  # Zone 1
        img.crop((width // 2, 0, width, height // 2)),  # Zone 2
        img.crop((0, height // 2, width // 2, height)),  # Zone 3
        img.crop((width // 2, height // 2, width, height))  # Zone 4
    ]

    # Extraire le nom du fichier sans l'extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Extraire le dossier de l'image
    folder = os.path.dirname(image_path)

    # Sauvegarder chaque morceau avec le suffixe correspondant dans le même dossier
    new_paths = []
    for i, cropped_img in enumerate(images_decoupees, 1):
        new_filename = os.path.join(folder, f"{base_name}_{i}.png")  # Utiliser .png pour les labels
        cropped_img.save(new_filename)
        print(f"Image découpée et sauvegardée sous: {new_filename}")
        new_paths.append(new_filename)

    return new_paths

def decouper_images_dans_json(json_file):
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Récupérer les chemins des images depuis les trois éléments train, test, validation
    new_data = {'train': [], 'test': [], 'validation': []}
    for category in ['train', 'test', 'validation']:
        for item in data[category]:
            # On prend seulement le 3ème élément de chaque sous-liste (les chemins des labels)
            label_path = item[2]  # Le chemin du label est le 3ème élément (index 2)

            # Appeler la fonction pour découper l'image de label
            new_label_paths = decouper_image(label_path)
            if new_label_paths:
                for new_label_path in new_label_paths:
                    new_item = item.copy()
                    new_item[2] = new_label_path
                    new_data[category].append(new_item)

    # Enregistrer le nouveau fichier JSON
    new_json_file = 'img_512_seg.json'  # Remplace par le chemin réel de ton nouveau fichier JSON
    with open(new_json_file, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"Nouveau fichier JSON enregistré sous: {new_json_file}")

# Exemple d'utilisation
json_file = 'img_1024_seg.json'  # Remplace par le chemin réel de ton fichier JSON
decouper_images_dans_json(json_file)
