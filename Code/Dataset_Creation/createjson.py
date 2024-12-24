import os
import shutil

def supprimer_sous_dossiers(base_path, suffix):
    for root, dirs, files in os.walk(base_path, topdown=False):
        for dir_name in dirs:
            if dir_name.endswith(suffix):
                dir_path = os.path.join(root, dir_name)
                print(f"Suppression du dossier : {dir_path}")
                shutil.rmtree(dir_path)

# Remplacez par le chemin de votre dossier principal
chemin_dossier_principal = "DynamicEarthNet/image_sat"
suffixe = "changement"

supprimer_sous_dossiers(chemin_dossier_principal, suffixe)
