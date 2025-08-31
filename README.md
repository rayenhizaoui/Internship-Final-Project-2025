# 🌽 Classification des Maladies du Maïs - IA

## 🎯 Description
Système d'intelligence artificielle pour la classification automatique des maladies du maïs utilisant des techniques de Deep Learning avancées.

### 🏆 Performances
- **8 modèles CNN/ViT** implémentés et comparés
- **Précision moyenne**: 85-95% selon le modèle
- **Déploiement web** avec interface utilisateur intuitive
- **Modèles quantifiés** pour inférence rapide (2-4x plus rapide)

## 🚀 Démarrage Rapide

### Installation
```bash
git clone https://github.com/rayenhizaoui/corn-disease-classification.git
cd corn-disease-classification
pip install -r requirements.txt
```

### Utilisation Immédiate
```bash
# Démarrer l'application web
python quick_start.py
# Choisir option 1: Application Web
```

## 📊 Modèles Disponibles

| Modèle | Architecture | Précision | Taille | Vitesse |
|--------|--------------|-----------|--------|---------|
| VGG16 | CNN Classique | ~87% | 500MB | Rapide |
| ResNet50 | Résiduel | ~91% | 100MB | Très Rapide |
| DenseNet121 | Dense | ~89% | 30MB | Ultra Rapide |
| MobileViT | Vision Transformer | ~93% | 20MB | Mobile |
| Ensemble | Combinaison | ~95% | Variable | Optimal |

## 🗂️ Structure du Projet

```
├── models/                 # Architectures des modèles
├── data/                   # Chargement et preprocessing
├── deployment/             # Application web Flask
├── utils/                  # Utilitaires et métriques
├── notebooks/              # Jupyter notebooks d'analyse
├── train_professional.py   # Entraînement complet
├── quick_start.py          # Interface de démarrage
└── requirements.txt        # Dépendances
```

## 📚 Documentation

- [Guide Complet des Commandes](GUIDE_COMMANDES_EXECUTION.md)
- [Structure Recommandée](STRUCTURE_PROJET_RECOMMANDEE.md)
- [Notebooks d'Analyse](notebooks/)

## ⚠️ Note sur les Données

Les datasets et modèles pré-entraînés ne sont pas inclus dans ce repository pour des raisons de taille. 

### Pour obtenir les données complètes:
1. **Dataset**: Téléchargez depuis [source] ou créez votre propre dataset
2. **Modèles pré-entraînés**: Disponibles dans les [GitHub Releases]
3. **Modèles quantifiés**: Générez avec `python quantize_production.py`

## 🛠️ Développement

### Entraîner vos propres modèles
```bash
python train_professional.py
```

### Optimiser pour production
```bash
python quantize_production.py
```

### Analyser les performances
```bash
python analyze_project_cleanup.py
```

## 📈 Résultats

- **Classification 4 classes**: Healthy, Northern Leaf Blight, Gray Leaf Spot, Northern Leaf Spot
- **Interface web responsive**: Upload d'images avec prédiction instantanée
- **Optimisation production**: Modèles quantifiés pour déploiement edge
- **Analyse complète**: Visualisations et métriques détaillées

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 👨‍💻 Auteur

**Rayen Hizaoui**
- GitHub: [@rayenhizaoui](https://github.com/rayenhizaoui)
- Projet: Stage Final 2025

---

⭐ N'hésitez pas à donner une étoile si ce projet vous a aidé !
