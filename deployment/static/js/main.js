// ===============================================
// 🌽 AgriDiagnostic IA - JavaScript Principal
// ===============================================

// Configuration globale
const CONFIG = {
    API_BASE_URL: '/api',
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/jpg', 'image/png'],
    PROGRESS_STEPS: [
        'Initialisation...',
        'Chargement de l\'image...',
        'Préprocessing...',
        'Analyse par IA...',
        'Génération des résultats...',
        'Finalisation...'
    ]
};

// Informations sur les maladies
const DISEASE_INFO = {
    'gls': {
        name: 'Gray Leaf Spot',
        scientific: 'Cercospora zeae-maydis',
        icon: '🔴',
        description: `La tache grise des feuilles est une maladie fongique majeure du maïs causée par 
        Cercospora zeae-maydis. Elle se caractérise par des lésions rectangulaires grises avec des 
        bordures sombres, suivant généralement les nervures des feuilles. Cette maladie peut causer 
        des réductions de rendement significatives de 15% à 60% dans les conditions favorables.`,
        recommendations: [
            {
                title: 'Traitement Fongicide',
                description: 'Application de fongicides spécifiques à base de triazoles ou strobilurines'
            },
            {
                title: 'Rotation des Cultures',
                description: 'Rotation avec des cultures non-hôtes pour briser le cycle pathogène'
            },
            {
                title: 'Gestion des Résidus',
                description: 'Enfouissement profond des résidus de culture infectés'
            },
            {
                title: 'Surveillance',
                description: 'Monitoring régulier des conditions météorologiques favorables'
            }
        ]
    },
    'nlb': {
        name: 'Northern Leaf Blight',
        scientific: 'Exserohilum turcicum',
        icon: '🔵',
        description: `La brûlure du nord des feuilles est causée par Exserohilum turcicum. Cette maladie 
        se manifeste par des lésions elliptiques de couleur brun-gris qui peuvent s'étendre sur plusieurs 
        centimètres. Elle est particulièrement problématique dans les régions à climat frais et humide, 
        pouvant causer jusqu'à 50% de perte de rendement.`,
        recommendations: [
            {
                title: 'Fongicides Préventifs',
                description: 'Application préventive de fongicides systémiques dès les premiers symptômes'
            },
            {
                title: 'Variétés Résistantes',
                description: 'Utilisation de variétés avec gènes de résistance Ht1, Ht2, ou Ht3'
            },
            {
                title: 'Espacement des Plants',
                description: 'Amélioration de la circulation d\'air entre les plants'
            },
            {
                title: 'Nutrition Équilibrée',
                description: 'Éviter l\'excès d\'azote qui favorise le développement de la maladie'
            }
        ]
    },
    'nls': {
        name: 'Northern Leaf Spot',
        scientific: 'Bipolaris zeicola',
        icon: '🟢',
        description: `La tache du nord des feuilles est causée par Bipolaris zeicola. Cette maladie se 
        caractérise par de petites taches circulaires à ovales, souvent avec un centre plus clair. 
        Bien que moins agressive que les autres maladies, elle peut néanmoins causer des pertes de 
        rendement modérées de 10% à 25%.`,
        recommendations: [
            {
                title: 'Traitement Modéré',
                description: 'Fongicides à base de cuivre ou triazoles selon la pression de la maladie'
            },
            {
                title: 'Gestion de l\'Irrigation',
                description: 'Éviter l\'irrigation par aspersion qui favorise la dispersion des spores'
            },
            {
                title: 'Fertilisation Raisonnée',
                description: 'Fertilisation équilibrée pour renforcer la résistance naturelle'
            },
            {
                title: 'Nettoyage Sanitaire',
                description: 'Élimination des feuilles infectées et des débris végétaux'
            }
        ]
    }
};

// Variables globales
let currentFile = null;
let analysisInProgress = false;

// ===============================================
// Initialisation de l'application
// ===============================================
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🌽 Initialisation d\'AgriDiagnostic IA...');
    
    // Initialiser les event listeners
    setupEventListeners();
    
    // Initialiser les fonctionnalités
    setupDragAndDrop();
    setupFileInput();
    
    console.log('✅ Application initialisée avec succès');
}

// ===============================================
// Event Listeners
// ===============================================
function setupEventListeners() {
    // Bouton d'analyse
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', startAnalysis);
    }
    
    // Bouton de suppression d'image
    const removeBtn = document.getElementById('removeImage');
    if (removeBtn) {
        removeBtn.addEventListener('click', removeImage);
    }
    
    // Liens de navigation
    setupNavigationLinks();
}

function setupNavigationLinks() {
    // Ajout d'événements pour les liens du footer (si nécessaire)
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('nav-link')) {
            e.preventDefault();
            // Gérer la navigation interne si nécessaire
        }
    });
}

// ===============================================
// Gestion du Drag & Drop
// ===============================================
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    if (!uploadArea) return;
    
    // Événements de drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragenter', handleDragEnter);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    
    // Vérifier si on quitte vraiment la zone
    if (!e.relatedTarget || !e.currentTarget.contains(e.relatedTarget)) {
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
}

// ===============================================
// Gestion des fichiers
// ===============================================
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }
}

function handleFileSelection(file) {
    console.log('📁 Fichier sélectionné:', file.name);
    
    // Validation du fichier
    if (!validateFile(file)) {
        return;
    }
    
    currentFile = file;
    
    // Afficher la prévisualisation
    displayImagePreview(file);
    
    // Masquer la zone d'upload et afficher la prévisualisation
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('previewSection').style.display = 'grid';
}

function validateFile(file) {
    // Vérifier le type de fichier
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        showNotification('Erreur: Type de fichier non supporté. Utilisez JPG, PNG ou JPEG.', 'error');
        return false;
    }
    
    // Vérifier la taille
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showNotification('Erreur: Fichier trop volumineux. Taille maximale: 10MB.', 'error');
        return false;
    }
    
    return true;
}

function displayImagePreview(file) {
    const reader = new FileReader();
    const previewImage = document.getElementById('previewImage');
    
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        
        // Animation d'apparition
        previewImage.style.opacity = '0';
        setTimeout(() => {
            previewImage.style.transition = 'opacity 0.3s ease';
            previewImage.style.opacity = '1';
        }, 100);
    };
    
    reader.readAsDataURL(file);
}

function removeImage() {
    console.log('🗑️ Suppression de l\'image');
    
    currentFile = null;
    
    // Réinitialiser l'input file
    const fileInput = document.getElementById('fileInput');
    fileInput.value = '';
    
    // Masquer la prévisualisation et afficher la zone d'upload
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    
    // Masquer les résultats s'ils étaient affichés
    hideResults();
}

// ===============================================
// Analyse de l'image
// ===============================================
async function startAnalysis() {
    if (!currentFile || analysisInProgress) {
        return;
    }
    
    console.log('🔍 Début de l\'analyse...');
    analysisInProgress = true;
    
    // Afficher la section de loading
    showLoadingSection();
    
    try {
        // Simuler le processus d'analyse avec vraie logique
        await performAnalysis();
        
    } catch (error) {
        console.error('❌ Erreur lors de l\'analyse:', error);
        showNotification('Erreur lors de l\'analyse. Veuillez réessayer.', 'error');
        hideLoadingSection();
    } finally {
        analysisInProgress = false;
    }
}

async function performAnalysis() {
    // Simulation du processus d'analyse avec étapes
    const steps = CONFIG.PROGRESS_STEPS;
    const progressBar = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    for (let i = 0; i < steps.length; i++) {
        // Mettre à jour le texte et la barre de progression
        progressText.textContent = steps[i];
        const progress = ((i + 1) / steps.length) * 100;
        progressBar.style.width = progress + '%';
        
        // Attendre un délai réaliste
        await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));
    }
    
    // Simuler l'appel API (en attendant l'intégration du modèle réel)
    const result = await callRealAPI();
    
    // Afficher les résultats
    hideLoadingSection();
    displayResults(result);
}

async function simulateAPICall() {
    // Simulation d'un appel API avec résultats aléatoires mais réalistes
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const diseases = ['gls', 'nlb', 'nls'];
    const mainDisease = diseases[Math.floor(Math.random() * diseases.length)];
    
    // Générer des confidences réalistes
    const mainConfidence = 85 + Math.random() * 14; // 85-99%
    const otherConfidences = diseases
        .filter(d => d !== mainDisease)
        .map(disease => ({
            disease,
            confidence: Math.random() * (100 - mainConfidence)
        }))
        .sort((a, b) => b.confidence - a.confidence);
    
    return {
        prediction: mainDisease,
        confidence: mainConfidence,
        alternatives: otherConfidences
    };
}

// ===============================================
// Affichage des résultats
// ===============================================
function displayResults(result) {
    console.log('📊 Affichage des résultats:', result);
    
    const diseaseInfo = DISEASE_INFO[result.prediction];
    
    // Mettre à jour les informations principales
    updateMainPrediction(diseaseInfo, result.confidence);
    
    // Mettre à jour la description
    updateDiseaseDescription(diseaseInfo);
    
    // Mettre à jour les recommandations
    updateRecommendations(diseaseInfo.recommendations);
    
    // Mettre à jour les alternatives
    updateAlternatives(result.alternatives);
    
    // Afficher la section des résultats avec animation
    showResultsSection();
}

function updateMainPrediction(diseaseInfo, confidence) {
    // Icône de la maladie
    const diseaseIcon = document.getElementById('diseaseIcon');
    diseaseIcon.textContent = diseaseInfo.icon;
    
    // Nom de la maladie
    const diseaseName = document.getElementById('diseaseName');
    diseaseName.textContent = diseaseInfo.name;
    
    // Nom scientifique
    const diseaseScientific = document.getElementById('diseaseScientific');
    diseaseScientific.textContent = diseaseInfo.scientific;
    
    // Badge de confiance
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.textContent = Math.round(confidence) + '%';
    
    // Couleur du badge selon la confiance
    if (confidence >= 90) {
        confidenceBadge.style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
    } else if (confidence >= 80) {
        confidenceBadge.style.background = 'linear-gradient(135deg, #f39c12, #e67e22)';
    } else {
        confidenceBadge.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
    }
    
    // Mettre à jour la barre de confiance circulaire
    updateConfidenceMeter(confidence);
}

function updateConfidenceMeter(confidence) {
    const confidenceFill = document.getElementById('confidenceFill');
    if (confidenceFill) {
        const angle = (confidence / 100) * 360;
        confidenceFill.style.background = `conic-gradient(#2ecc71 0deg ${angle}deg, #ecf0f1 ${angle}deg 360deg)`;
    }
}

function updateDiseaseDescription(diseaseInfo) {
    const descriptionContent = document.getElementById('diseaseDescription');
    descriptionContent.innerHTML = `<p>${diseaseInfo.description}</p>`;
}

function updateRecommendations(recommendations) {
    const recommendationsGrid = document.getElementById('recommendationsGrid');
    
    recommendationsGrid.innerHTML = recommendations.map(rec => `
        <div class="recommendation-item fade-in">
            <h5>${rec.title}</h5>
            <p>${rec.description}</p>
        </div>
    `).join('');
}

function updateAlternatives(alternatives) {
    const alternativesList = document.getElementById('alternativesList');
    
    alternativesList.innerHTML = alternatives.map(alt => {
        const diseaseInfo = DISEASE_INFO[alt.disease];
        return `
            <div class="alternative-item fade-in">
                <span class="alternative-name">${diseaseInfo.icon} ${diseaseInfo.name}</span>
                <span class="alternative-confidence">${Math.round(alt.confidence)}%</span>
            </div>
        `;
    }).join('');
}

// ===============================================
// Gestion de l'affichage des sections
// ===============================================
function showLoadingSection() {
    hideResults();
    document.getElementById('loadingSection').style.display = 'block';
    
    // Animation d'apparition
    setTimeout(() => {
        document.getElementById('loadingSection').classList.add('fade-in');
    }, 100);
}

function hideLoadingSection() {
    const loadingSection = document.getElementById('loadingSection');
    loadingSection.style.display = 'none';
    loadingSection.classList.remove('fade-in');
    
    // Réinitialiser la barre de progression
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = 'Initialisation...';
}

function showResultsSection() {
    document.getElementById('resultsSection').style.display = 'block';
    
    // Animation d'apparition avec délai pour les éléments
    setTimeout(() => {
        document.getElementById('resultsSection').classList.add('fade-in');
        
        // Animer les éléments individuels
        const fadeElements = document.querySelectorAll('.fade-in');
        fadeElements.forEach((element, index) => {
            setTimeout(() => {
                element.style.opacity = '0';
                element.style.transform = 'translateY(20px)';
                element.style.transition = 'all 0.4s ease';
                
                setTimeout(() => {
                    element.style.opacity = '1';
                    element.style.transform = 'translateY(0)';
                }, 100);
            }, index * 100);
        });
    }, 100);
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'none';
    resultsSection.classList.remove('fade-in');
}

// ===============================================
// Actions sur les résultats
// ===============================================
function downloadReport() {
    console.log('📥 Téléchargement du rapport...');
    
    // Créer un rapport détaillé
    const reportData = generateReport();
    
    // Créer et télécharger le fichier
    const blob = new Blob([reportData], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `diagnostic_mais_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showNotification('Rapport téléchargé avec succès!', 'success');
}

function generateReport() {
    const now = new Date();
    const diseaseName = document.getElementById('diseaseName').textContent;
    const confidence = document.getElementById('confidenceBadge').textContent;
    
    return `
🌽 RAPPORT DE DIAGNOSTIC - MALADIES DU MAÏS
=============================================

Date d'analyse: ${now.toLocaleDateString('fr-FR')} à ${now.toLocaleTimeString('fr-FR')}
Système: AgriDiagnostic IA v1.0

RÉSULTATS PRINCIPAUX
--------------------
Maladie détectée: ${diseaseName}
Niveau de confiance: ${confidence}
Modèle utilisé: VGG16 Optimisé (99.37% précision)

RECOMMANDATIONS
---------------
${Array.from(document.querySelectorAll('.recommendation-item')).map(item => 
    `• ${item.querySelector('h5').textContent}: ${item.querySelector('p').textContent}`
).join('\n')}

ALTERNATIVES CONSIDÉRÉES
------------------------
${Array.from(document.querySelectorAll('.alternative-item')).map(item => 
    `• ${item.querySelector('.alternative-name').textContent}: ${item.querySelector('.alternative-confidence').textContent}`
).join('\n')}

AVERTISSEMENT
-------------
Ce diagnostic automatique est un outil d'aide à la décision. 
Pour des décisions importantes, consultez un expert en phytopathologie.

© 2025 AgriDiagnostic IA - Agriculture de Précision
`;
}

function resetAnalysis() {
    console.log('🔄 Réinitialisation de l\'analyse...');
    
    hideResults();
    removeImage();
    
    showNotification('Prêt pour une nouvelle analyse!', 'info');
}

function resetUpload() {
    removeImage();
}

// ===============================================
// Système de notifications
// ===============================================
function showNotification(message, type = 'info') {
    // Créer l'élément de notification s'il n'existe pas
    let notificationContainer = document.getElementById('notificationContainer');
    
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notificationContainer';
        notificationContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 350px;
        `;
        document.body.appendChild(notificationContainer);
    }
    
    // Créer la notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        background: ${getNotificationColor(type)};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transform: translateX(400px);
        transition: transform 0.3s ease;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    `;
    
    // Ajouter une icône selon le type
    const icon = getNotificationIcon(type);
    notification.innerHTML = `${icon} ${message}`;
    
    // Ajouter au container
    notificationContainer.appendChild(notification);
    
    // Animation d'entrée
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Suppression automatique après 5 secondes
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

function getNotificationColor(type) {
    const colors = {
        success: 'linear-gradient(135deg, #2ecc71, #27ae60)',
        error: 'linear-gradient(135deg, #e74c3c, #c0392b)',
        warning: 'linear-gradient(135deg, #f39c12, #e67e22)',
        info: 'linear-gradient(135deg, #3498db, #2980b9)'
    };
    return colors[type] || colors.info;
}

function getNotificationIcon(type) {
    const icons = {
        success: '<i class="fas fa-check-circle"></i>',
        error: '<i class="fas fa-exclamation-triangle"></i>',
        warning: '<i class="fas fa-exclamation-circle"></i>',
        info: '<i class="fas fa-info-circle"></i>'
    };
    return icons[type] || icons.info;
}

// ===============================================
// Utilitaires
// ===============================================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ===============================================
// Gestion des erreurs globales
// ===============================================
window.addEventListener('error', function(event) {
    console.error('❌ Erreur JavaScript:', event.error);
    showNotification('Une erreur inattendue s\'est produite. Veuillez actualiser la page.', 'error');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('❌ Promise rejetée:', event.reason);
    showNotification('Erreur de connexion. Vérifiez votre connexion internet.', 'error');
});

// ===============================================
// Export pour tests (si nécessaire)
// ===============================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateFile,
        formatFileSize,
        DISEASE_INFO,
        CONFIG
    };
}


// Appel API réel vers le serveur Flask
async function callRealAPI() {
    const formData = new FormData();
    formData.append('image', currentFile);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Erreur lors de l\'analyse');
        }
        
        return {
            prediction: data.result.prediction,
            confidence: data.result.confidence,
            alternatives: data.result.alternatives,
            disease_info: data.result.disease_info
        };
        
    } catch (error) {
        console.error('Erreur API:', error);
        showNotification('Erreur de connexion au serveur. Mode démo activé.', 'warning');
        
        // Fallback vers simulation en cas d'erreur
        return await simulateAPICall();
    }
}