#!/usr/bin/env python3
"""
📥 TÉLÉCHARGEUR DE MODÈLES QUANTIFIÉS
Télécharge automatiquement les modèles depuis GitHub Releases
"""

import requests
import os
from pathlib import Path
import json
from tqdm import tqdm

class ModelsDownloader:
    def __init__(self):
        self.base_url = "https://github.com/rayenhizaoui/Internship-Final-Project-2025/releases/download/v1.0"
        self.models_dir = Path("quantized_models_all")
        self.models_dir.mkdir(exist_ok=True)
        
        # Modèles disponibles
        self.available_models = {
            "deit3_quantized.pth": {"size_mb": 84.5, "description": "DeiT3 Vision Transformer"},
            "densenet121_quantized.pth": {"size_mb": 28.1, "description": "DenseNet121 CNN"},
            "maxvit_quantized.pth": {"size_mb": 54.6, "description": "MaxViT hybrid model"},
            "mvitv2_quantized.pth": {"size_mb": 49.8, "description": "MViTv2 Vision Transformer"},
            "resnet50_quantized.pth": {"size_mb": 92.0, "description": "ResNet50 CNN"},
            "vgg16_quantized.pth": {"size_mb": 174.1, "description": "VGG16 CNN"},
            "xception_quantized.pth": {"size_mb": 79.7, "description": "Xception CNN"}
        }
    
    def download_file(self, filename, description):
        """Télécharge un fichier avec barre de progression"""
        url = f"{self.base_url}/{filename}"
        file_path = self.models_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} déjà téléchargé")
            return True
        
        print(f"📥 Téléchargement: {description}")
        print(f"🔗 URL: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Téléchargé: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur téléchargement {filename}: {str(e)}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def download_all(self):
        """Télécharge tous les modèles"""
        print("🚀 TÉLÉCHARGEMENT DES MODÈLES QUANTIFIÉS")
        print("="*50)
        
        success_count = 0
        total_count = len(self.available_models)
        
        for filename, info in self.available_models.items():
            if self.download_file(filename, info["description"]):
                success_count += 1
        
        print(f"\n📊 RÉSULTAT: {success_count}/{total_count} modèles téléchargés")
        
        if success_count == total_count:
            print("🎉 Tous les modèles ont été téléchargés avec succès !")
        elif success_count > 0:
            print("⚠️  Certains modèles n'ont pas pu être téléchargés")
        else:
            print("❌ Aucun modèle n'a pu être téléchargé")
        
        return success_count
    
    def download_specific(self, model_names):
        """Télécharge des modèles spécifiques"""
        print(f"📥 Téléchargement de {len(model_names)} modèles spécifiques")
        
        success_count = 0
        for model_name in model_names:
            if model_name in self.available_models:
                info = self.available_models[model_name]
                if self.download_file(model_name, info["description"]):
                    success_count += 1
            else:
                print(f"❌ Modèle non trouvé: {model_name}")
        
        return success_count
    
    def list_available(self):
        """Liste les modèles disponibles"""
        print("📋 MODÈLES QUANTIFIÉS DISPONIBLES")
        print("="*40)
        
        for filename, info in self.available_models.items():
            local_path = self.models_dir / filename
            status = "✅ Téléchargé" if local_path.exists() else "📥 Disponible"
            print(f"{status}: {filename} ({info['size_mb']:.1f} MB)")
            print(f"    📝 {info['description']}")
        
        total_size = sum(info['size_mb'] for info in self.available_models.values())
        downloaded_size = sum(
            info['size_mb'] for filename, info in self.available_models.items()
            if (self.models_dir / filename).exists()
        )
        
        print(f"\n📊 Total disponible: {total_size:.1f} MB")
        print(f"📊 Déjà téléchargé: {downloaded_size:.1f} MB")

def main():
    """Interface utilisateur"""
    downloader = ModelsDownloader()
    
    print("🤖 GESTIONNAIRE DE MODÈLES QUANTIFIÉS")
    print("="*40)
    print("1. Lister les modèles disponibles")
    print("2. Télécharger tous les modèles")
    print("3. Télécharger des modèles spécifiques")
    print("4. Télécharger les modèles légers seulement")
    
    choice = input("\n🔢 Votre choix (1-4): ").strip()
    
    if choice == "1":
        downloader.list_available()
    
    elif choice == "2":
        downloader.download_all()
    
    elif choice == "3":
        downloader.list_available()
        models = input("\n📝 Entrez les noms des modèles (séparés par des virgules): ").strip()
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if model_list:
            downloader.download_specific(model_list)
        else:
            print("❌ Aucun modèle spécifié")
    
    elif choice == "4":
        # Modèles < 50MB
        light_models = [
            name for name, info in downloader.available_models.items()
            if info['size_mb'] < 50
        ]
        print(f"📥 Téléchargement des modèles légers: {light_models}")
        downloader.download_specific(light_models)
    
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()
