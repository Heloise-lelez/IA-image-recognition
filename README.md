# 🐾 Metazoa AI

**Metazoa AI** est une application web combinant une interface HTML/CSS simple et une intelligence artificielle entraînée avec **PyTorch** pour détecter automatiquement certaines espèces animales à partir d'une image.  
Le projet vise à rendre l’identification d’un animal accessible à tous, directement depuis un navigateur.

## 🚀 Fonctionnalités

- 🌍 Interface web épurée (HTML/CSS)
- 🧠 Modèle IA de détection d'animaux entraîné avec PyTorch, exporté en ONNX
- 📸 Téléversement d’images via l’interface utilisateur
- 🐾 Reconnaissance de plusieurs espèces (ex : lynx, tortue, salamandre…)
- 📊 Résultats affichés en temps réel avec étiquette et probabilité

## 🛠️ Technologies utilisées

- **Frontend** : HTML5, CSS3, JavaScript
- **Backend IA** : Python 3, PyTorch, ONNX
- **Modèle d’inférence Web** : [ONNX Runtime Web](https://onnxruntime.ai/docs/build/web.html)

## 🔧 Installation et exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/Heloise-lelez/IA-image-recognition
cd metazoa-ai
```

### 2. (Optionnel) Créer un environnement virtuel Python

```bash
python -m venv venv
source venv/bin/activate   # Sur Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer un serveur local pour la démo web

Dans le dossier contenant `index.html` :

```bash
python -m http.server 8000
```

Puis, ouvre ton navigateur à l’adresse :  
[http://localhost:8000](http://localhost:8000)

---

## 📌 Remarques

- Le modèle IA a été exporté au format `.onnx` pour permettre une inférence directement dans le navigateur, sans backend Python.
- Le modèle reconnaît actuellement un ensemble limité d'espèces animales. Pour l'étendre, tu peux réentraîner le modèle avec d'autres données.

---

## 📷 Exemple de résultat

> Une image est téléversée, et l’application affiche immédiatement :
> **Prédiction : Salamandre (92.3%)**

---

## 🧪 Dépendances principales (requirements.txt)

```txt
torch
torchvision
onnx
onnxruntime
numpy
Pillow
```

