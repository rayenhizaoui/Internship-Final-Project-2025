#!/usr/bin/env python3
"""
Script de démarrage simple pour AgriDiagnostic IA
"""

import os
import sys

def start_app():
    """Démarre l'application web"""
    print("=" * 60)
    print("🌽 AGRIDIAGNOSTIC IA - DÉMARRAGE")
    print("=" * 60)
    print("Application Web de Classification des Maladies du Maïs")
    print("Précision: 99.37% | Classes: GLS, NLB, NLS")
    print("=" * 60)
    
    # Changer vers le répertoire de l'application
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Démarrer l'application Flask
    try:
        print("\n🚀 Démarrage du serveur web...")
        print("📍 URL: http://127.0.0.1:5000")
        print("💡 Appuyez sur Ctrl+C pour arrêter")
        print("-" * 60)
        
        # Importer et démarrer l'app
        sys.path.append('deployment')
        from deployment import app as flask_app
        
        flask_app.app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Application arrêtée par l'utilisateur")
        print("Merci d'avoir utilisé AgriDiagnostic IA!")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("Consultez le guide de déploiement pour plus d'aide.")

if __name__ == "__main__":
    start_app()
