# Projet PIR2024 🚀

Le projet **PIR2024** regroupe plusieurs outils et scripts dédiés au traitement d'images 🖼️, à l'intelligence artificielle 🤖 et à la gestion de jeux de données 📊. Il est conçu pour des applications de classification et de segmentation.

Les données proviennent d'un Dataset réduit de **DynamicEarthNet** 🌍.

## Structure du Projet 📁

### 1. **Scripts Principaux** 📝
- **`Application.py`**  
  Interface Utilisateur (UI). Elle affiche une fenêtre Tkinter permettant de modifier les paramètres de notre script **`complet.py`** et **`complet_multi.py`** avec les options suivantes :
  - Type d'encodeur 🧑‍💻
  - Poids appliqué si les données sont déséquilibrées ⚖️
  - Nombre d'epochs ⏳
  - Fonction de perte 📉
  - Répartition du Dataset 🔢
  - Enregistrement du modèle et des métriques associées 📈
  - Choix entre segmentation binaire ou sémantique 🧩

  Voir l'image ci-dessous 👇.

- **`complet.py`**  
  Ce script implémente un modèle **Unet** de deep learning pour le traitement d'images. Il est conçu pour entraîner et évaluer le modèle sur le jeu de données fourni dans le dossier **`Repartition_Dataset`**.

- **`complet_Multi.py`**  
  Variante avancée de **`complet.py`**, ce script permet de faire de la détection de changement **sémantique** 🔍.

### 2. **Gestion des Données** 📦
Les données nécessaires à l'exécution du projet ne sont pas incluses dans ce dépôt pour des raisons de taille et de confidentialité 🔒. Elles sont disponibles via un lien spécifique fourni séparément. Une fois les données téléchargées, il est essentiel de les placer à la **racine du projet** pour garantir le bon fonctionnement des scripts.

### 3. **Préparation et Exécution** 🔧

#### Préparation de l'environnement 🖥️
- Assurez-vous d'avoir **Python 3.11** installé. 🐍
- Installez les dépendances requises avec `pip` ou `conda`.

#### Placer les données 📂
- Téléchargez les données via le lien fourni.  
- Déplacez-les à la racine du projet pour que les scripts puissent y accéder correctement.

#### Exécution des scripts 🎬
- Exécutez **`Application.py`** pour démarrer l'interface utilisateur.
- Entraînez un modèle avec un jeu de données spécifique.
- Entraînez et évaluez le modèle sur plusieurs jeux de données simultanément.
