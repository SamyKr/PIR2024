import os
import numpy as np
import matplotlib.pyplot as plt

def plot_f1_scores(directory):
    """
    Recherche dans le dossier tous les fichiers avec l'extension `.npy`,
    vérifie si les résultats contiennent des données valides et
    trace les courbes F1 en fonction des epochs avec les noms de modèles en légende.
    """
    all_results_files = []

    # Recherche des fichiers avec l'extension `.npy` dans le dossier
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "results.npy":
                all_results_files.append(os.path.join(root, file))

    # Initialisation du graphique
    plt.figure(figsize=(12, 6))
    curves_added = False  # Pour vérifier si au moins une courbe est ajoutée

    for file_path in all_results_files:
        try:
            # Chargement des données
            loaded_results = np.load(file_path, allow_pickle=True).item()

            # Validation de la structure des données
            if "data" not in loaded_results or "results" not in loaded_results:
                print(f"Fichier ignoré (clés manquantes): {file_path}")
                continue

            # Extraction des informations principales
            model_name = loaded_results["data"].get("model_name", "Unknown Model")
            results = loaded_results["results"]

            # Extraction des métriques
            f1_scores = [result["f1"] for result in results if "f1" in result and "epoch" in result]
            epochs = [result["epoch"] for result in results if "f1" in result and "epoch" in result]

            if len(f1_scores) == len(epochs) and len(f1_scores) > 0:
                # Tracé de la courbe
                plt.plot(epochs, f1_scores, marker='o', linestyle='-', label=model_name)
                curves_added = True
            else:
                print(f"Fichier ignoré (données invalides): {file_path}")

        except Exception as e:
            print(f"Erreur lors du traitement de {file_path}: {e}")

    # Personnalisation du graphique
    if curves_added:
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.title("F1 Scores par Epoch pour différents modèles")
        plt.legend(title="Model Info", loc="upper left", fontsize=10)
        plt.grid(True)
        plt.show()
    else:
        print("Aucune courbe valide à afficher.")

# Utilisation
# Remplacez 'votre_dossier' par le chemin vers votre dossier
plot_f1_scores("bon_segmentation")
