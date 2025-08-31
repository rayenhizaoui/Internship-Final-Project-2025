# 📦 Instructions GitHub Releases - Modèles Volumineux

## 🎯 Objectif
Les modèles quantifiés >25MB sont hébergés sur **GitHub Releases** pour contourner les limitations de taille des repositories.

## 🚀 Étapes pour Créer la Release

### 1️⃣ Accéder aux Releases
```
🔗 https://github.com/rayenhizaoui/Internship-Final-Project-2025/releases
```

### 2️⃣ Créer une Nouvelle Release
- Cliquer sur **"Create a new release"**
- **Tag**: `v1.0`
- **Title**: `🤖 Quantized Models Release v1.0`
- **Description**: Voir ci-dessous

### 3️⃣ Description de la Release
```markdown
# 🤖 Modèles Quantifiés - Classification Maladies du Maïs

Cette release contient les modèles quantifiés optimisés pour la production.

## 📊 Modèles Inclus

| Modèle | Taille | Description |
|--------|--------|-------------|
| `deit3_quantized.pth` | 84.5 MB | DeiT3 Vision Transformer quantized |
| `densenet121_quantized.pth` | 28.1 MB | DenseNet121 CNN quantized |
| `maxvit_quantized.pth` | 54.6 MB | MaxViT hybrid model quantized |
| `mvitv2_quantized.pth` | 49.8 MB | MViTv2 Vision Transformer quantized |
| `resnet50_quantized.pth` | 92.0 MB | ResNet50 CNN quantized |
| `vgg16_quantized.pth` | 174.1 MB | VGG16 CNN quantized |
| `xception_quantized.pth` | 79.7 MB | Xception CNN quantized |

## 🔧 Utilisation

### Téléchargement Automatique
```bash
python download_models.py
```

### Chargement des Modèles
```python
import torch
model = torch.jit.load('quantized_models_all/mobilevit_quantized.pth')
model.eval()
```

## 📈 Performance
- **Réduction de taille**: ~75%
- **Précision préservée**: 95%+
- **Optimisation**: Dynamic INT8 Quantization

## 🔗 Repository Principal
[Internship-Final-Project-2025](https://github.com/rayenhizaoui/Internship-Final-Project-2025)
```

### 4️⃣ Upload des Modèles
Glisser-déposer les fichiers suivants depuis `quantized_models_all/`:
- deit3_quantized.pth
- densenet121_quantized.pth
- maxvit_quantized.pth
- mvitv2_quantized.pth
- resnet50_quantized.pth
- vgg16_quantized.pth
- xception_quantized.pth

### 5️⃣ Publier la Release
- Cocher **"Set as the latest release"**
- Cliquer sur **"Publish release"**

## ✅ Vérification
Après publication, vérifier que:
- Tous les modèles sont téléchargeables
- Les URLs correspondent au script `download_models.py`
- La documentation est claire

## 🔗 URLs Finales
Les modèles seront accessibles à:
```
https://github.com/rayenhizaoui/Internship-Final-Project-2025/releases/download/v1.0/[nom_modele].pth
```

## 💡 Avantages
- ✅ Contourne la limite GitHub de 100MB
- ✅ Téléchargement séparé et optionnel
- ✅ Versioning des modèles
- ✅ Statistiques de téléchargement
- ✅ Repository principal reste léger
