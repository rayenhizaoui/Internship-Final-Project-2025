#!/usr/bin/env python3
"""
ANALYSE COMPLÈTE DU PROJET POUR NETTOYAGE
========================================
Script d'analyse pour identifier tous les fichiers inutiles, redondants 
et non nécessaires dans le projet de classification des maladies du maïs.

Objectif: Créer un projet propre et organisé
Date: 2025-08-31
"""

import os
import glob
import json
from pathlib import Path
from collections import defaultdict

def get_dir_size(path):
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

def format_size(size_bytes):
    """Formater la taille en MB"""
    return round(size_bytes / (1024 * 1024), 2)

def analyze_project_structure():
    """Analyser la structure complète du projet"""
    
    print("🔍 ANALYSE COMPLÈTE DU PROJET")
    print("=" * 60)
    
    analysis_results = {
        'total_files': 0,
        'total_size_mb': 0,
        'categories': {
            'CORE_ESSENTIAL': [],
            'MODELS_ORIGINAL': [],
            'QUANTIZED_REDUNDANT': [],
            'TEST_DEBUG_FILES': [],
            'VISUALIZATION_EXCESS': [],
            'LOGS_TEMP': [],
            'DEPLOYMENT': [],
            'DUPLICATES': [],
            'USELESS': []
        },
        'directories_analysis': {}
    }
    
    # Scanner tous les fichiers
    all_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                all_files.append({
                    'path': file_path,
                    'name': file,
                    'dir': root,
                    'size': size,
                    'ext': os.path.splitext(file)[1].lower()
                })
                analysis_results['total_files'] += 1
                analysis_results['total_size_mb'] += size
            except:
                continue
    
    analysis_results['total_size_mb'] = format_size(analysis_results['total_size_mb'])
    
    print(f"📊 Fichiers totaux: {analysis_results['total_files']}")
    print(f"📊 Taille totale: {analysis_results['total_size_mb']} MB")
    
    # Catégoriser les fichiers
    for file_info in all_files:
        path = file_info['path'].replace('\\', '/')
        name = file_info['name']
        ext = file_info['ext']
        
        # Fichiers essentiels du core
        if any(essential in path for essential in [
            'data/data_loader.py', 'data/preprocessing.py',
            'models/vgg16_model.py', 'models/resnet50_model.py', 'models/densenet121_model.py',
            'utils/config.py', 'utils/metrics.py', 'requirements.txt'
        ]):
            analysis_results['categories']['CORE_ESSENTIAL'].append(file_info)
        
        # Modèles originaux (garder les meilleurs seulement)
        elif 'results/' in path and ext == '.pth':
            analysis_results['categories']['MODELS_ORIGINAL'].append(file_info)
        
        # Modèles quantifiés redondants
        elif 'quantized_models' in path:
            analysis_results['categories']['QUANTIZED_REDUNDANT'].append(file_info)
        
        # Fichiers de test/debug
        elif any(test_prefix in name for test_prefix in [
            'test_', 'debug_', 'quick_', 'check_', 'view_', 'inspect_'
        ]):
            analysis_results['categories']['TEST_DEBUG_FILES'].append(file_info)
        
        # Visualisations excessives
        elif any(viz_path in path for viz_path in [
            'visualizations/', 'training_curves/', 'confusion_matrices/', 'metrics/'
        ]):
            analysis_results['categories']['VISUALIZATION_EXCESS'].append(file_info)
        
        # Fichiers de déploiement
        elif any(deploy_path in path for deploy_path in [
            'deployment/', 'templates/', 'static/'
        ]) or name in ['app.py', 'deploy.py', 'launch_app.py']:
            analysis_results['categories']['DEPLOYMENT'].append(file_info)
        
        # Logs et fichiers temporaires
        elif any(temp_indicator in path for temp_indicator in [
            '.log', '.tmp', '__pycache__', '.cache', '.pyc'
        ]) or ext in ['.log', '.tmp', '.pyc']:
            analysis_results['categories']['LOGS_TEMP'].append(file_info)
        
        # Fichiers potentiellement inutiles
        elif any(useless_name in name for useless_name in [
            'backup', 'old', 'copy', 'temp', 'tmp'
        ]) or name.startswith('.'):
            analysis_results['categories']['USELESS'].append(file_info)
    
    return analysis_results

def analyze_quantized_redundancy(analysis_results):
    """Analyser la redondance dans les modèles quantifiés"""
    
    print(f"\n🔍 ANALYSE DES MODÈLES QUANTIFIÉS")
    print("-" * 40)
    
    quantized_dirs = defaultdict(list)
    
    for file_info in analysis_results['categories']['QUANTIZED_REDUNDANT']:
        dir_name = file_info['dir'].replace('.\\', '').replace('./', '')
        quantized_dirs[dir_name].append(file_info)
    
    total_quantized_size = 0
    recommendations = []
    
    for dir_name, files in quantized_dirs.items():
        dir_size = sum(f['size'] for f in files)
        total_quantized_size += dir_size
        
        print(f"📁 {dir_name}:")
        print(f"   - Fichiers: {len(files)}")
        print(f"   - Taille: {format_size(dir_size)} MB")
        
        # Recommandations
        if 'quantized_models_all' in dir_name:
            recommendations.append(f"✅ GARDER: {dir_name} (le plus complet)")
        else:
            recommendations.append(f"🗑️  SUPPRIMER: {dir_name} (redondant)")
    
    print(f"\n📊 Total modèles quantifiés: {format_size(total_quantized_size)} MB")
    print(f"💡 Espace économisable: ~{format_size(total_quantized_size * 0.75)} MB")
    
    return recommendations

def create_cleanup_recommendations(analysis_results):
    """Créer les recommandations de nettoyage"""
    
    print(f"\n🎯 RECOMMANDATIONS DE NETTOYAGE")
    print("=" * 60)
    
    recommendations = {
        'critical_delete': [],
        'safe_delete': [],
        'optional_delete': [],
        'keep_essential': []
    }
    
    # Analyse par catégorie
    categories = analysis_results['categories']
    
    # 1. Modèles quantifiés redondants (CRITIQUE)
    quantized_dirs = set()
    for file_info in categories['QUANTIZED_REDUNDANT']:
        dir_name = file_info['dir']
        quantized_dirs.add(dir_name)
    
    print(f"🔴 PRIORITÉ CRITIQUE - Modèles quantifiés redondants:")
    for dir_name in sorted(quantized_dirs):
        files_in_dir = [f for f in categories['QUANTIZED_REDUNDANT'] if f['dir'] == dir_name]
        total_size = sum(f['size'] for f in files_in_dir)
        
        if 'quantized_models_all' not in dir_name:
            print(f"   🗑️  {dir_name} ({format_size(total_size)} MB)")
            recommendations['critical_delete'].append(dir_name)
        else:
            print(f"   ✅ {dir_name} (GARDER - {format_size(total_size)} MB)")
            recommendations['keep_essential'].append(dir_name)
    
    # 2. Fichiers de test/debug (SÉCURISÉ)
    print(f"\n🟡 PRIORITÉ HAUTE - Fichiers de test/debug ({len(categories['TEST_DEBUG_FILES'])} fichiers):")
    test_size = sum(f['size'] for f in categories['TEST_DEBUG_FILES'])
    print(f"   📊 Taille totale: {format_size(test_size)} MB")
    
    for file_info in categories['TEST_DEBUG_FILES'][:5]:  # Afficher 5 premiers
        print(f"   🗑️  {file_info['path']} ({format_size(file_info['size'])} MB)")
    
    if len(categories['TEST_DEBUG_FILES']) > 5:
        print(f"   ... et {len(categories['TEST_DEBUG_FILES'])-5} autres fichiers")
    
    recommendations['safe_delete'].extend([f['path'] for f in categories['TEST_DEBUG_FILES']])
    
    # 3. Visualisations excessives (OPTIONNEL)
    print(f"\n🟢 PRIORITÉ MOYENNE - Visualisations excessives ({len(categories['VISUALIZATION_EXCESS'])} fichiers):")
    viz_size = sum(f['size'] for f in categories['VISUALIZATION_EXCESS'])
    print(f"   📊 Taille totale: {format_size(viz_size)} MB")
    
    # Garder les essentiels
    essential_viz = [
        'complete_training_dashboard.png',
        'dataset_comprehensive_analysis.png', 
        'CNN_models_comparison.png',
        'index.html'
    ]
    
    for file_info in categories['VISUALIZATION_EXCESS']:
        if not any(essential in file_info['name'] for essential in essential_viz):
            recommendations['optional_delete'].append(file_info['path'])
    
    # 4. Logs et temporaires
    print(f"\n🟢 PRIORITÉ FAIBLE - Logs et temporaires ({len(categories['LOGS_TEMP'])} fichiers):")
    logs_size = sum(f['size'] for f in categories['LOGS_TEMP'])
    print(f"   📊 Taille totale: {format_size(logs_size)} MB")
    recommendations['safe_delete'].extend([f['path'] for f in categories['LOGS_TEMP']])
    
    return recommendations

def calculate_savings(recommendations):
    """Calculer l'espace qui sera libéré"""
    
    total_savings = 0
    
    # Calculer pour les dossiers critiques
    for dir_path in recommendations['critical_delete']:
        if os.path.exists(dir_path):
            total_savings += get_dir_size(dir_path)
    
    # Calculer pour les fichiers sécurisés
    for file_path in recommendations['safe_delete']:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                total_savings += os.path.getsize(file_path)
            elif os.path.isdir(file_path):
                total_savings += get_dir_size(file_path)
    
    # Calculer pour les fichiers optionnels
    for file_path in recommendations['optional_delete']:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                total_savings += os.path.getsize(file_path)
    
    return format_size(total_savings)

def main():
    """Fonction principale d'analyse"""
    
    print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE DU PROJET")
    print("🎯 Objectif: Identifier les fichiers inutiles pour un projet propre")
    print("=" * 70)
    
    # Analyse complète
    analysis_results = analyze_project_structure()
    
    # Analyse de la redondance quantifiée
    quantized_recommendations = analyze_quantized_redundancy(analysis_results)
    
    # Recommandations de nettoyage
    recommendations = create_cleanup_recommendations(analysis_results)
    
    # Calcul des économies
    estimated_savings = calculate_savings(recommendations)
    
    # Résumé final
    print(f"\n" + "🎉" * 70)
    print("📊 RÉSUMÉ FINAL DE L'ANALYSE")
    print("🎉" * 70)
    print(f"📈 Fichiers totaux analysés: {analysis_results['total_files']}")
    print(f"📊 Taille actuelle du projet: {analysis_results['total_size_mb']} MB")
    print(f"💾 Espace libérable estimé: ~{estimated_savings} MB")
    print(f"📉 Réduction estimée: ~{(float(estimated_savings)/float(analysis_results['total_size_mb'])*100):.1f}%")
    
    print(f"\n🎯 ACTIONS RECOMMANDÉES:")
    print(f"   🔴 CRITIQUE: Supprimer {len(recommendations['critical_delete'])} dossiers redondants")
    print(f"   🟡 SÉCURISÉ: Supprimer {len(recommendations['safe_delete'])} fichiers de test/logs")
    print(f"   🟢 OPTIONNEL: Supprimer {len(recommendations['optional_delete'])} visualisations excessives")
    
    # Sauvegarder l'analyse
    analysis_report = {
        'analysis_date': '2025-08-31',
        'project_stats': {
            'total_files': analysis_results['total_files'],
            'total_size_mb': analysis_results['total_size_mb'],
            'estimated_savings_mb': estimated_savings
        },
        'recommendations': recommendations,
        'categories_summary': {cat: len(files) for cat, files in analysis_results['categories'].items()}
    }
    
    with open('project_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport détaillé sauvegardé: project_analysis_report.json")
    print(f"✅ Analyse terminée! Prochaine étape: Créer les scripts de nettoyage")

if __name__ == "__main__":
    main()
