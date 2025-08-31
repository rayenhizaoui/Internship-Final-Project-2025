#!/usr/bin/env python3
"""
SCRIPT DE NETTOYAGE INTERACTIF ET SÉCURISÉ
=========================================
Script pour nettoyer le projet de manière sélective et sécurisée.
Basé sur l'analyse complète du projet.

🎯 OBJECTIF: Réduire la taille du projet de 7.08 GB à ~2-3 GB
💾 ÉCONOMIE ESTIMÉE: ~1.8 GB

Date: 2025-08-31
"""

import os
import shutil
import json
from pathlib import Path
import time

class ProjectCleaner:
    """Nettoyeur de projet interactif et sécurisé"""
    
    def __init__(self):
        self.deleted_files = 0
        self.space_saved = 0
        self.backup_created = False
        
    def get_dir_size(self, path):
        """Calculer la taille d'un dossier"""
        total = 0
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total += os.path.getsize(file_path)
        except:
            pass
        return total
    
    def format_size(self, size_bytes):
        """Formater la taille"""
        if size_bytes >= 1024**3:  # GB
            return f"{size_bytes / (1024**3):.2f} GB"
        elif size_bytes >= 1024**2:  # MB
            return f"{size_bytes / (1024**2):.2f} MB"
        else:  # KB
            return f"{size_bytes / 1024:.2f} KB"
    
    def safe_delete(self, path):
        """Suppression sécurisée avec vérification"""
        try:
            if os.path.isfile(path):
                size = os.path.getsize(path)
                os.remove(path)
                self.deleted_files += 1
                self.space_saved += size
                print(f"   ✅ Fichier supprimé: {path} ({self.format_size(size)})")
                return True
            elif os.path.isdir(path):
                size = self.get_dir_size(path)
                shutil.rmtree(path)
                self.deleted_files += 1
                self.space_saved += size
                print(f"   ✅ Dossier supprimé: {path} ({self.format_size(size)})")
                return True
            else:
                print(f"   ⚠️  Non trouvé: {path}")
                return False
        except Exception as e:
            print(f"   ❌ Erreur lors de la suppression de {path}: {e}")
            return False
    
    def show_main_menu(self):
        """Afficher le menu principal"""
        print("\n" + "🧹" * 60)
        print("🧹  NETTOYAGE INTERACTIF DU PROJET")
        print("🧹" * 60)
        print()
        print("📊 ÉTAT ACTUEL DU PROJET:")
        print(f"   - Taille totale: ~7.08 GB")
        print(f"   - Fichiers totaux: 3,396")
        print(f"   - Espace libérable: ~1.8 GB")
        print()
        print("🎯 OPTIONS DE NETTOYAGE:")
        print()
        print("1️⃣  🔴 CRITIQUE - Modèles quantifiés redondants (~1.8 GB)")
        print("    Supprime les dossiers de modèles quantifiés en double")
        print("    Garde seulement 'quantized_models_all'")
        print()
        print("2️⃣  🟡 SÉCURISÉ - Fichiers de test/debug (~30 MB)")
        print("    Supprime test_*.py, debug_*.py, quick_*.py, etc.")
        print()
        print("3️⃣  🟢 OPTIONNEL - Visualisations excessives (~11 MB)")
        print("    Garde les visualisations essentielles seulement")
        print()
        print("4️⃣  🟢 LOGS - Fichiers temporaires et logs (~1 MB)")
        print("    Supprime *.log, __pycache__, etc.")
        print()
        print("5️⃣  🚀 NETTOYAGE COMPLET (Recommandé)")
        print("    Applique toutes les optimisations ci-dessus")
        print()
        print("6️⃣  📊 Voir le détail de chaque catégorie")
        print("7️⃣  🚪 Quitter sans rien faire")
        print()
    
    def clean_quantized_redundant(self):
        """Nettoyer les modèles quantifiés redondants"""
        print("\n🔴 NETTOYAGE - MODÈLES QUANTIFIÉS REDONDANTS")
        print("-" * 50)
        
        # Dossiers à supprimer (garder seulement quantized_models_all)
        redundant_dirs = [
            "quantized_models_final",
            "quantized_models_final_optimized", 
            "quantized_models_production",
            "quantized_models_ultra"
        ]
        
        total_size_before = 0
        for dir_name in redundant_dirs:
            if os.path.exists(dir_name):
                size = self.get_dir_size(dir_name)
                total_size_before += size
                print(f"📁 {dir_name}: {self.format_size(size)}")
        
        print(f"\n💾 Espace total à libérer: {self.format_size(total_size_before)}")
        print(f"✅ Dossier conservé: quantized_models_all")
        
        confirm = input(f"\n⚠️  Confirmez-vous la suppression? (oui/non): ").lower()
        
        if confirm == 'oui':
            print(f"\n🚀 Suppression en cours...")
            for dir_name in redundant_dirs:
                if os.path.exists(dir_name):
                    self.safe_delete(dir_name)
            
            print(f"✅ Nettoyage terminé!")
            return True
        else:
            print("❌ Opération annulée")
            return False
    
    def clean_test_debug_files(self):
        """Nettoyer les fichiers de test/debug"""
        print("\n🟡 NETTOYAGE - FICHIERS DE TEST/DEBUG")
        print("-" * 50)
        
        import glob
        
        # Patterns des fichiers à supprimer
        test_patterns = [
            "test_*.py", "debug_*.py", "quick_*.py", "check_*.py",
            "inspect_*.py", "view_*.py"
        ]
        
        files_to_delete = []
        total_size = 0
        
        for pattern in test_patterns:
            for file_path in glob.glob(pattern):
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    files_to_delete.append(file_path)
                    total_size += size
                    print(f"📄 {file_path}: {self.format_size(size)}")
        
        print(f"\n📊 {len(files_to_delete)} fichiers à supprimer")
        print(f"💾 Espace à libérer: {self.format_size(total_size)}")
        
        if files_to_delete:
            confirm = input(f"\n⚠️  Confirmez-vous la suppression? (oui/non): ").lower()
            
            if confirm == 'oui':
                print(f"\n🚀 Suppression en cours...")
                for file_path in files_to_delete:
                    self.safe_delete(file_path)
                print(f"✅ Nettoyage terminé!")
                return True
            else:
                print("❌ Opération annulée")
                return False
        else:
            print("ℹ️  Aucun fichier de test trouvé")
            return False
    
    def clean_visualizations(self):
        """Nettoyer les visualisations excessives"""
        print("\n🟢 NETTOYAGE - VISUALISATIONS EXCESSIVES")
        print("-" * 50)
        
        # Visualisations essentielles à garder
        essential_viz = [
            'complete_training_dashboard.png',
            'dataset_comprehensive_analysis.png',
            'CNN_models_comparison.png',
            'index.html'
        ]
        
        viz_dirs = ['visualizations', 'training_curves', 'confusion_matrices', 'metrics']
        files_to_delete = []
        total_size = 0
        
        for viz_dir in viz_dirs:
            if os.path.exists(viz_dir):
                for root, dirs, files in os.walk(viz_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if not any(essential in file for essential in essential_viz):
                            size = os.path.getsize(file_path)
                            files_to_delete.append(file_path)
                            total_size += size
                            print(f"📄 {file_path}: {self.format_size(size)}")
        
        print(f"\n📊 {len(files_to_delete)} fichiers à supprimer")
        print(f"💾 Espace à libérer: {self.format_size(total_size)}")
        print(f"✅ Fichiers conservés: {len(essential_viz)} visualisations essentielles")
        
        if files_to_delete:
            confirm = input(f"\n⚠️  Confirmez-vous la suppression? (oui/non): ").lower()
            
            if confirm == 'oui':
                print(f"\n🚀 Suppression en cours...")
                for file_path in files_to_delete:
                    self.safe_delete(file_path)
                print(f"✅ Nettoyage terminé!")
                return True
            else:
                print("❌ Opération annulée")
                return False
        else:
            print("ℹ️  Aucune visualisation excessive trouvée")
            return False
    
    def clean_logs_temp(self):
        """Nettoyer les logs et fichiers temporaires"""
        print("\n🟢 NETTOYAGE - LOGS ET TEMPORAIRES")
        print("-" * 50)
        
        import glob
        
        temp_patterns = ["*.log", "*.tmp", "__pycache__", "*.pyc"]
        files_to_delete = []
        total_size = 0
        
        for pattern in temp_patterns:
            if pattern == "__pycache__":
                # Chercher les dossiers __pycache__
                for root, dirs, files in os.walk("."):
                    if "__pycache__" in dirs:
                        pycache_path = os.path.join(root, "__pycache__")
                        size = self.get_dir_size(pycache_path)
                        files_to_delete.append(pycache_path)
                        total_size += size
                        print(f"📁 {pycache_path}: {self.format_size(size)}")
            else:
                for file_path in glob.glob(pattern):
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        files_to_delete.append(file_path)
                        total_size += size
                        print(f"📄 {file_path}: {self.format_size(size)}")
        
        print(f"\n📊 {len(files_to_delete)} éléments à supprimer")
        print(f"💾 Espace à libérer: {self.format_size(total_size)}")
        
        if files_to_delete:
            confirm = input(f"\n⚠️  Confirmez-vous la suppression? (oui/non): ").lower()
            
            if confirm == 'oui':
                print(f"\n🚀 Suppression en cours...")
                for file_path in files_to_delete:
                    self.safe_delete(file_path)
                print(f"✅ Nettoyage terminé!")
                return True
            else:
                print("❌ Opération annulée")
                return False
        else:
            print("ℹ️  Aucun fichier temporaire trouvé")
            return False
    
    def complete_cleanup(self):
        """Nettoyage complet du projet"""
        print("\n🚀 NETTOYAGE COMPLET DU PROJET")
        print("=" * 60)
        print("⚠️  ATTENTION: Cette opération va supprimer:")
        print("   🔴 Tous les modèles quantifiés redondants (~1.8 GB)")
        print("   🟡 Tous les fichiers de test/debug")
        print("   🟢 Les visualisations excessives")
        print("   🟢 Les logs et fichiers temporaires")
        print()
        print("✅ CONSERVERA:")
        print("   📁 quantized_models_all/ (modèles quantifiés optimaux)")
        print("   📁 results/ (modèles originaux)")
        print("   📁 data/ (données et preprocessing)")
        print("   📁 models/ (architectures)")
        print("   📁 deployment/ (application web)")
        print("   📄 Visualisations essentielles")
        print()
        
        final_confirm = input("🤔 Voulez-vous vraiment procéder au nettoyage complet? (OUI/non): ")
        
        if final_confirm.upper() == 'OUI':
            print("\n🚀 DÉBUT DU NETTOYAGE COMPLET...")
            start_time = time.time()
            
            operations = [
                ("🔴 Modèles quantifiés redondants", self.clean_quantized_redundant),
                ("🟡 Fichiers de test/debug", self.clean_test_debug_files),
                ("🟢 Visualisations excessives", self.clean_visualizations),
                ("🟢 Logs et temporaires", self.clean_logs_temp)
            ]
            
            for operation_name, operation_func in operations:
                print(f"\n{operation_name}...")
                operation_func()
            
            duration = time.time() - start_time
            
            print(f"\n" + "🎉" * 60)
            print("🎉  NETTOYAGE COMPLET TERMINÉ!")
            print("🎉" * 60)
            print(f"📊 Statistiques:")
            print(f"   - Éléments supprimés: {self.deleted_files}")
            print(f"   - Espace libéré: {self.format_size(self.space_saved)}")
            print(f"   - Durée: {duration:.2f} secondes")
            print(f"   - Taille estimée finale: ~3-4 GB (vs 7.08 GB initial)")
            
            return True
        else:
            print("❌ Nettoyage complet annulé")
            return False
    
    def show_details(self):
        """Afficher les détails de chaque catégorie"""
        print("\n📊 DÉTAILS DES CATÉGORIES")
        print("=" * 50)
        
        # Charger le rapport d'analyse s'il existe
        try:
            with open('project_analysis_report.json', 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print("📈 Statistiques générales:")
            stats = report['project_stats']
            print(f"   - Fichiers totaux: {stats['total_files']:,}")
            print(f"   - Taille totale: {stats['total_size_mb']} MB")
            print(f"   - Économie estimée: {stats['estimated_savings_mb']} MB")
            
            print("\n📋 Répartition par catégorie:")
            categories = report['categories_summary']
            for category, count in categories.items():
                print(f"   - {category}: {count} fichiers")
                
        except:
            print("⚠️  Rapport d'analyse non trouvé. Exécutez d'abord analyze_project_cleanup.py")
    
    def run(self):
        """Lancer le nettoyeur interactif"""
        
        while True:
            self.show_main_menu()
            
            choice = input("🎯 Choisissez une option (1-7): ").strip()
            
            if choice == '1':
                self.clean_quantized_redundant()
            elif choice == '2':
                self.clean_test_debug_files()
            elif choice == '3':
                self.clean_visualizations()
            elif choice == '4':
                self.clean_logs_temp()
            elif choice == '5':
                self.complete_cleanup()
                break  # Sortir après nettoyage complet
            elif choice == '6':
                self.show_details()
            elif choice == '7':
                print("\n👋 Au revoir! Aucune modification effectuée.")
                break
            else:
                print("\n❌ Option invalide. Veuillez choisir entre 1 et 7.")
            
            if choice in ['1', '2', '3', '4']:
                continue_choice = input("\n🤔 Voulez-vous continuer le nettoyage? (oui/non): ")
                if continue_choice.lower() != 'oui':
                    break
        
        # Résumé final
        if self.deleted_files > 0:
            print(f"\n📊 RÉSUMÉ FINAL:")
            print(f"   ✅ Éléments supprimés: {self.deleted_files}")
            print(f"   💾 Espace libéré: {self.format_size(self.space_saved)}")
            print(f"   🎯 Votre projet est maintenant plus propre et organisé!")

def main():
    """Fonction principale"""
    print("🧹 LANCEMENT DU NETTOYEUR DE PROJET")
    print("🎯 Optimisation du projet de classification des maladies du maïs")
    print("=" * 70)
    
    cleaner = ProjectCleaner()
    cleaner.run()

if __name__ == "__main__":
    main()
