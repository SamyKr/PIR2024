import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def browse_directory(entry):
    directory = filedialog.askdirectory()
    if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

def browse_file(entry):
    filename = filedialog.askopenfilename()
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

def add_model():
    selected_model = model_name_var.get()
    num_epochs = num_epochs_entry.get()
    pos_weight = pos_weight_entry.get()
    loss_function = loss_function_var.get()
    segmentation_type = segmentation_type_var.get()

    # Créer un nom unique pour le répertoire de sauvegarde du modèle et des métriques
    pos_weight_str = "true" if pos_weight else "false"
    save_dir = f"{selected_model}_{num_epochs}_{pos_weight_str}_{loss_function}_{segmentation_type}"

    # Récupérer le répertoire de sauvegarde fourni par l'utilisateur
    base_save_dir = model_result_entry.get()

    # Ajouter un sous-dossier avec la structure model_name_epoch_pos_weight_loss_function_segmentation_type
    full_save_dir = os.path.join(base_save_dir, save_dir)

    # Créer le répertoire si nécessaire
    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir)

    models_to_test.append({
        "model": selected_model,
        "epochs": num_epochs,
        "pos_weight": pos_weight_str,
        "loss_function": loss_function,
        "segmentation_type": segmentation_type,
        "save_dir": full_save_dir
    })

    update_model_list()

def update_model_list():
    models_listbox.delete(0, tk.END)
    for model in models_to_test:
        models_listbox.insert(tk.END, f"{model['model']} - Epochs: {model['epochs']} - Pos Weight: {model['pos_weight']} - Loss Function: {model['loss_function']} - Segmentation Type: {model['segmentation_type']}")

def delete_model():
    selected_index = models_listbox.curselection()
    if selected_index:
        index = selected_index[0]
        del models_to_test[index]
        update_model_list()

def run_program():
    num_epochs = num_epochs_entry.get()
    pos_weight = pos_weight_entry.get()
    input_json = input_json_entry.get()

    # Ajouter des noms de fichiers aux répertoires
    for model_data in models_to_test:
        model_name = model_data["model"]
        save_dir = model_data["save_dir"]
        loss_function = model_data["loss_function"]
        segmentation_type = model_data["segmentation_type"]

        # Créer les chemins complets
        model_result_path = os.path.join(save_dir, "model.pth")
        metrics_result_path = os.path.join(save_dir, "results.npy")

        # Choisir le script en fonction du type de segmentation
        if segmentation_type == "binary":
            script_path = "Code/entrainement_test_validation_pos_weight.py"
        elif segmentation_type == "multi":
            script_path = "Code/entrainement_test_validation_multi.py"
        else:
            raise ValueError(f"Unknown segmentation type: {segmentation_type}")

        # Construire la commande
        command = [
            "python", script_path,  # Remplacez par le nom de votre script
            f"--model_name={model_name}",
            f"--num_epochs={model_data['epochs']}",
            f"--input_json={input_json}",
            f"--model_result={model_result_path}",
            f"--metrics_result={metrics_result_path}",
            f"--loss_function={loss_function}",
        ]

        # Si pos_weight n'est pas vide, l'ajouter à la commande
        if pos_weight:
            command.append(f"--pos_weight={pos_weight}")

        # Afficher un message "En cours..." pendant l'exécution
        output_text.insert(tk.END, f"En cours pour le modèle {model_name}...\n")
        output_text.yview(tk.END)
        root.update_idletasks()  # Mettre à jour l'interface pour afficher "En cours..."

        # Exécuter la commande et capturer la sortie en temps réel
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Lire la sortie standard ligne par ligne
            for stdout_line in iter(process.stdout.readline, ""):
                if "return F.conv2d" not in stdout_line:  # Ignorer les lignes contenant 'return F.conv2d'
                    if "UserWarning" not in stdout_line:  # Filtrer les avertissements de type UserWarning
                        output_text.insert(tk.END, stdout_line)
                        output_text.yview(tk.END)  # Faire défiler le texte automatiquement
                        root.update_idletasks()  # Mettre à jour l'interface graphique pour afficher la nouvelle ligne

            # Lire les erreurs standard s'il y en a
            for stderr_line in iter(process.stderr.readline, ""):
                if "return F.conv2d" not in stderr_line:  # Ignorer les lignes contenant 'return F.conv2d'
                    if "UserWarning" not in stderr_line:  # Filtrer les avertissements de type UserWarning
                        output_text.insert(tk.END, stderr_line)
                        output_text.yview(tk.END)
                        root.update_idletasks()

            process.stdout.close()
            process.stderr.close()
            process.wait()  # Attendre que le processus se termine

            # Une fois le processus terminé, mettre à jour la sortie
            output_text.insert(tk.END, f"Fin de l'exécution pour le modèle {model_name}.\n")
            output_text.yview(tk.END)
            root.update_idletasks()  # Mettre à jour l'interface

        except subprocess.CalledProcessError as e:
            output_text.insert(tk.END, e.stderr)

def on_segmentation_type_change():
    segmentation_type = segmentation_type_var.get()
    if segmentation_type == "multi":
        loss_function_var.set("DiceLoss")
        loss_function_menu.config(state="disabled")
    else:
        loss_function_menu.config(state="normal")

# Créer la fenêtre principale
root = tk.Tk()
root.title("Change Detection - Paramètres")

# Créer les champs de saisie des paramètres
frame = tk.Frame(root)
frame.pack(pady=10, padx=10)

# Liste des modèles disponibles
model_names = ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b1", "vgg16", "mobilenet_v2"]

# Variable pour l'OptionMenu
model_name_var = tk.StringVar(root)
model_name_var.set(model_names[0])  # Initialiser la valeur par défaut

# Label et OptionMenu pour model_name
tk.Label(frame, text="Model Name:").grid(row=0, column=0, sticky="e")
model_name_menu = tk.OptionMenu(frame, model_name_var, *model_names)
model_name_menu.grid(row=0, column=1)

tk.Label(frame, text="Number of Epochs:").grid(row=1, column=0, sticky="e")
num_epochs_entry = tk.Entry(frame)
num_epochs_entry.grid(row=1, column=1)
num_epochs_entry.insert(0, "4")

tk.Label(frame, text="Pos Weight:").grid(row=2, column=0, sticky="e")
pos_weight_entry = tk.Entry(frame)
pos_weight_entry.grid(row=2, column=1)

# Ajouter un OptionMenu pour la fonction de pertes
loss_functions = ["CrossEntropyLoss", "BCEWithLogitsLoss", "MSELoss", "DiceLoss"]
loss_function_var = tk.StringVar(root)
loss_function_var.set(loss_functions[0])  # Initialiser la valeur par défaut

tk.Label(frame, text="Loss Function:").grid(row=3, column=0, sticky="e")
loss_function_menu = tk.OptionMenu(frame, loss_function_var, *loss_functions)
loss_function_menu.grid(row=3, column=1)

# Ajouter un Radiobutton pour le type de segmentation
segmentation_type_var = tk.StringVar(value="multi")

tk.Label(frame, text="Segmentation Type:").grid(row=4, column=0, sticky="e")
tk.Radiobutton(frame, text="Binary", variable=segmentation_type_var, value="binary", command=on_segmentation_type_change).grid(row=4, column=1, sticky="w")
tk.Radiobutton(frame, text="Multi-class", variable=segmentation_type_var, value="multi", command=on_segmentation_type_change).grid(row=4, column=2, sticky="w")

# JSON Input
input_json_label = tk.Label(frame, text="Split data JSON format:")
input_json_label.grid(row=5, column=0, sticky="e")
input_json_entry = tk.Entry(frame, width=40)
input_json_entry.grid(row=5, column=1)
input_json_entry.insert(0, "img_1024.json")
browse_json_button = tk.Button(frame, text="Browse", command=lambda: browse_file(input_json_entry))
browse_json_button.grid(row=5, column=2)

# Ajouter un bouton pour ajouter un modèle
add_model_button = tk.Button(root, text="Add Model", command=add_model, bg="blue", fg="white")
add_model_button.pack(pady=10)

# Liste des modèles ajoutés
models_to_test = []

# Liste box pour afficher les modèles ajoutés
models_listbox = tk.Listbox(root, height=5, width=50)
models_listbox.pack(pady=10)

# Ajouter un bouton pour supprimer un modèle
delete_model_button = tk.Button(root, text="Delete Model", command=delete_model, bg="red", fg="white")
delete_model_button.pack(pady=10)

# Model Result
model_result_label = tk.Label(frame, text="Save Directory:")
model_result_label.grid(row=6, column=0, sticky="e")
model_result_entry = tk.Entry(frame, width=40)
model_result_entry.grid(row=6, column=1)
model_result_entry.insert(0, "./poubelle")
browse_model_button = tk.Button(frame, text="Browse", command=lambda: browse_directory(model_result_entry))
browse_model_button.grid(row=6, column=2)

# Ajouter un bouton pour lancer le script
run_button = tk.Button(root, text="Run", command=run_program, bg="green", fg="white")
run_button.pack(pady=10)

# Zone pour afficher la sortie
output_text = tk.Text(root, height=15, width=80)
output_text.pack(pady=10)

# Appeler la fonction pour initialiser l'état du menu déroulant de la fonction de perte
on_segmentation_type_change()

# Lancer la boucle principale
root.mainloop()
