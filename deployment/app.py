#!/usr/bin/env python3
"""
🌽 AgriDiagnostic IA - Serveur Flask pour le déploiement
Application web pour la classification des maladies du maïs
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment/logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Imports Flask
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Imports pour l'IA
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Configuration de l'application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'corn_disease_classification_2025'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'deployment/uploads'
app.config['STATIC_FOLDER'] = 'deployment/static'
app.config['TEMPLATES_FOLDER'] = 'deployment/templates'

# Configuration du modèle
MODEL_CONFIG = {
    'model_path': 'vgg16_optimized_model.pth',
    'classes': ['gls', 'nlb', 'nls'],
    'class_names': {
        'gls': 'Gray Leaf Spot',
        'nlb': 'Northern Leaf Blight', 
        'nls': 'Northern Leaf Spot'
    },
    'input_size': (224, 224),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Informations détaillées sur les maladies
DISEASE_INFO = {
    'gls': {
        'name': 'Gray Leaf Spot',
        'scientific': 'Cercospora zeae-maydis',
        'icon': '🔴',
        'severity': 'Élevée',
        'impact': '15-60% de réduction du rendement',
        'symptoms': 'Lésions rectangulaires grises avec bordures sombres',
        'description': '''La tache grise des feuilles est une maladie fongique majeure du maïs causée par 
        Cercospora zeae-maydis. Elle se caractérise par des lésions rectangulaires grises avec des 
        bordures sombres, suivant généralement les nervures des feuilles. Cette maladie peut causer 
        des réductions de rendement significatives de 15% à 60% dans les conditions favorables.''',
        'recommendations': [
            'Application de fongicides spécifiques à base de triazoles ou strobilurines',
            'Rotation avec des cultures non-hôtes pour briser le cycle pathogène', 
            'Enfouissement profond des résidus de culture infectés',
            'Monitoring régulier des conditions météorologiques favorables'
        ]
    },
    'nlb': {
        'name': 'Northern Leaf Blight',
        'scientific': 'Exserohilum turcicum',
        'icon': '🔵',
        'severity': 'Élevée',
        'impact': 'Jusqu\'à 50% de perte de rendement',
        'symptoms': 'Lésions elliptiques de couleur brun-gris',
        'description': '''La brûlure du nord des feuilles est causée par Exserohilum turcicum. Cette maladie 
        se manifeste par des lésions elliptiques de couleur brun-gris qui peuvent s'étendre sur plusieurs 
        centimètres. Elle est particulièrement problématique dans les régions à climat frais et humide, 
        pouvant causer jusqu'à 50% de perte de rendement.''',
        'recommendations': [
            'Application préventive de fongicides systémiques dès les premiers symptômes',
            'Utilisation de variétés avec gènes de résistance Ht1, Ht2, ou Ht3',
            'Amélioration de la circulation d\'air entre les plants',
            'Éviter l\'excès d\'azote qui favorise le développement de la maladie'
        ]
    },
    'nls': {
        'name': 'Northern Leaf Spot',
        'scientific': 'Bipolaris zeicola',
        'icon': '🟢',
        'severity': 'Modérée',
        'impact': '10-25% de perte de rendement',
        'symptoms': 'Petites taches circulaires à ovales',
        'description': '''La tache du nord des feuilles est causée par Bipolaris zeicola. Cette maladie se 
        caractérise par de petites taches circulaires à ovales, souvent avec un centre plus clair. 
        Bien que moins agressive que les autres maladies, elle peut néanmoins causer des pertes de 
        rendement modérées de 10% à 25%.''',
        'recommendations': [
            'Fongicides à base de cuivre ou triazoles selon la pression de la maladie',
            'Éviter l\'irrigation par aspersion qui favorise la dispersion des spores',
            'Fertilisation équilibrée pour renforcer la résistance naturelle',
            'Élimination des feuilles infectées et des débris végétaux'
        ]
    }
}

# Variables globales pour le modèle
model = None
transform = None
device = torch.device(MODEL_CONFIG['device'])

# ===============================================
# Initialisation de l'application
# ===============================================

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        'deployment/uploads',
        'deployment/logs',
        'deployment/static/css',
        'deployment/static/js',
        'deployment/templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Répertoire créé/vérifié: {directory}")

def load_model():
    """Charge le modèle pré-entraîné"""
    global model, transform
    
    try:
        logger.info("🤖 Chargement du modèle...")
        
        # Vérifier si le modèle existe
        if not os.path.exists(MODEL_CONFIG['model_path']):
            logger.error(f"❌ Modèle non trouvé: {MODEL_CONFIG['model_path']}")
            return False
        
        # Charger le modèle
        model = torch.load(MODEL_CONFIG['model_path'], map_location=device)
        model.eval()
        model.to(device)
        
        # Définir les transformations
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ Modèle chargé avec succès sur {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
        logger.error(traceback.format_exc())
        return False

def init_app():
    """Initialise l'application"""
    logger.info("🌽 Initialisation d'AgriDiagnostic IA...")
    
    # Créer les répertoires
    create_directories()
    
    # Charger le modèle
    if not load_model():
        logger.warning("⚠️ Application démarrée sans modèle (mode démo)")
    
    logger.info("✅ Application initialisée avec succès")

# ===============================================
# Routes principales
# ===============================================

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Vérification de l'état de l'application"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'device': str(device),
        'version': '1.0.0'
    }
    return jsonify(status)

# ===============================================
# API d'analyse
# ===============================================

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyse une image pour détecter les maladies"""
    try:
        # Vérifier qu'un fichier a été envoyé
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Aucun fichier fourni'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Aucun fichier sélectionné'
            }), 400
        
        # Valider le fichier
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Type de fichier non supporté'
            }), 400
        
        # Sauvegarder temporairement le fichier
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"📁 Fichier sauvegardé: {filepath}")
        
        # Analyser l'image
        if model is not None:
            result = predict_disease(filepath)
        else:
            # Mode démo si le modèle n'est pas chargé
            result = simulate_prediction()
        
        # Supprimer le fichier temporaire
        try:
            os.remove(filepath)
        except:
            pass
        
        logger.info(f"✅ Analyse terminée: {result['prediction']} ({result['confidence']:.2f}%)")
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'error': 'Fichier trop volumineux (max 10MB)'
        }), 413
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Erreur interne du serveur'
        }), 500

def allowed_file(filename):
    """Vérifie si le type de fichier est autorisé"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(image_path):
    """Prédit la maladie à partir d'une image"""
    try:
        # Charger et préprocesser l'image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item() * 100
        
        # Classe prédite
        predicted_class = MODEL_CONFIG['classes'][predicted_class_idx]
        
        # Alternatives (autres classes avec leurs probabilités)
        alternatives = []
        for i, class_name in enumerate(MODEL_CONFIG['classes']):
            if i != predicted_class_idx:
                alternatives.append({
                    'disease': class_name,
                    'confidence': probabilities[i].item() * 100
                })
        
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Construire le résultat complet
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'alternatives': alternatives,
            'disease_info': DISEASE_INFO[predicted_class]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        return simulate_prediction()  # Fallback vers simulation

def simulate_prediction():
    """Simule une prédiction pour les tests/démo"""
    import random
    
    # Sélection aléatoire d'une maladie
    predicted_class = random.choice(MODEL_CONFIG['classes'])
    confidence = random.uniform(85, 99)
    
    # Générer des alternatives
    alternatives = []
    remaining_classes = [c for c in MODEL_CONFIG['classes'] if c != predicted_class]
    remaining_prob = 100 - confidence
    
    for i, class_name in enumerate(remaining_classes):
        alt_confidence = random.uniform(0, remaining_prob / len(remaining_classes))
        alternatives.append({
            'disease': class_name,
            'confidence': alt_confidence
        })
    
    alternatives.sort(key=lambda x: x['confidence'], reverse=True)
    
    result = {
        'prediction': predicted_class,
        'confidence': confidence,
        'alternatives': alternatives,
        'disease_info': DISEASE_INFO[predicted_class],
        'demo_mode': True
    }
    
    return result

# ===============================================
# Routes statiques
# ===============================================

@app.route('/static/<path:filename>')
def static_files(filename):
    """Sert les fichiers statiques"""
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

# ===============================================
# Gestion des erreurs
# ===============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Ressource non trouvée'
    }), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'success': False,
        'error': 'Fichier trop volumineux (max 10MB)'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Erreur interne: {error}")
    return jsonify({
        'success': False,
        'error': 'Erreur interne du serveur'
    }), 500

# ===============================================
# Point d'entrée
# ===============================================

if __name__ == '__main__':
    # Initialiser l'application
    init_app()
    
    # Configuration du serveur
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '127.0.0.1')
    
    logger.info(f"🚀 Démarrage du serveur sur {host}:{port}")
    logger.info(f"🔧 Mode debug: {debug_mode}")
    logger.info(f"🤖 Modèle chargé: {model is not None}")
    
    # Démarrer le serveur
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        threaded=True
    )
