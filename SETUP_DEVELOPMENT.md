# SETUP POUR DÉVELOPPEMENT COMPLET

## 📥 Récupérer les Données Complètes

Ce repository GitHub contient seulement le code source optimisé.
Pour le développement complet, vous devez récupérer:

### 1. Dataset (2-3 GB)
```bash
# Option A: Depuis GitHub LFS (si configuré)
git lfs pull

# Option B: Téléchargement manuel
# Placez le dataset dans: data/Dataset/
```

### 2. Modèles Pré-entraînés (1-2 GB)
```bash
# Télécharger depuis GitHub Releases
# Ou entraîner vos propres modèles:
python train_professional.py
```

### 3. Génération des Modèles Quantifiés
```bash
# Après avoir les modèles originaux:
python quantize_production.py
```

## 🔧 Configuration Développement

### Environnement Python
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Structure Complète
Après setup, vous devriez avoir:
```
├── data/Dataset/           # Données d'entraînement
├── results/               # Modèles entraînés
├── quantized_models_all/  # Modèles optimisés
├── visualizations/        # Graphiques générés
└── logs/                  # Logs d'entraînement
```

## ⚡ Démarrage Rapide Post-Setup
```bash
python quick_start.py
```
