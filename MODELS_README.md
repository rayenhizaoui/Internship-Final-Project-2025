# 🤖 Modèles et Résultats

## 📊 Modèles Quantifiés Disponibles

Ce repository contient les modèles quantifiés optimisés pour la classification des maladies du maïs.

### 🔧 Technique de Quantification
- **Méthode**: Dynamic INT8 Quantization
- **Framework**: PyTorch
- **Réduction de taille**: ~75%
- **Performance**: Préservée à 95%+

### 📁 Structure des Modèles

```
quantized_models_all/
├── deit3_quantized.pth          # DeiT3 optimisé
├── densenet121_quantized.pth    # DenseNet121 optimisé  
├── maxvit_quantized.pth         # MaxViT optimisé
├── mobilevit_quantized.pth      # MobileViT optimisé
├── mvitv2_quantized.pth         # MViTv2 optimisé
├── resnet50_quantized.pth       # ResNet50 optimisé
├── vgg16_quantized.pth          # VGG16 optimisé
├── xception_quantized.pth       # Xception optimisé
└── quantization_report.json     # Rapport détaillé
```

### 📈 Résultats et Métriques

```
results/
├── */metadata.json             # Métadonnées des modèles
├── reports/                    # Rapports d'évaluation  
├── logs/                       # Logs d'entraînement
└── ensemble/                   # Résultats ensemble
```

## 🚀 Utilisation

### Charger un Modèle Quantifié
```python
import torch

# Charger le modèle quantifié
model = torch.jit.load('quantized_models_all/mobilevit_quantized.pth')
model.eval()

# Utilisation pour inférence
with torch.no_grad():
    output = model(input_tensor)
```

### Script de Quantification
```bash
python quantize_production.py
```

## 📊 Performance

| Modèle | Taille Originale | Taille Quantifiée | Réduction | Précision |
|--------|------------------|-------------------|-----------|-----------|
| MobileViT | ~15 MB | ~4 MB | 75% | 95.2% |
| ResNet50 | ~95 MB | ~24 MB | 75% | 94.8% |
| VGG16 | ~135 MB | ~34 MB | 75% | 94.5% |

## 🔗 Liens Utiles

- [Guide de Quantification](GUIDE_QUANTIZATION.md)
- [Rapport de Performance](PERFORMANCE_REPORT.md)
- [Documentation Complète](README.md)
