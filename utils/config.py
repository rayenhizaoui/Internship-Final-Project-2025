# Configuration settings for the corn disease classification project
import os

# Dataset Configuration
DATASET_DIR = r"C:\Users\rayen\Desktop\corn-disease-classification - Copie (7)\Dataset\Dataset_Original"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Global Training Configuration
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 25  # Full training epochs
LEARNING_RATE = 3e-4

# Model Configuration
N_CLASSES = 3
MODEL_SAVE_PATH = "results/ensemble/final_ensemble_model.pth"

# Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 7
ENABLE_EARLY_STOPPING = True

# Random Seeds Configuration (CORRECTION DES RÉSULTATS IDENTIQUES)
MODEL_SPECIFIC_SEEDS = {
    'vgg16': 42,
    'resnet50': 123,
    'densenet121': 456,
    'xception': 789,
    'mobilevit': 321,
    'maxvit': 654,
    'mvitv2': 987,
    'deit3': 147
}

# Validation Split Configuration (CORRECCIÓN)
VALIDATION_TEST_SIZE = 0.25  # Changé de 0.2 à 0.25 pour éviter les résultats identiques

# Random Search Configuration (Optimized for efficiency)
RANDOM_SEARCH_ITERATIONS = 8  # Reduced for time efficiency
RANDOM_SEARCH_CV_FOLDS = 3
ENABLE_RANDOM_SEARCH = True
RANDOM_SEARCH_EPOCHS = 5  # Short epochs for hyperparameter search

# GPU Optimization
GPU_CONFIG = {
    'enable_mixed_precision': True,
    'max_memory_fraction': 0.9,
    'gradient_accumulation_steps': 2
}

# Models Architecture Configuration
MODEL_CATEGORIES = {
    'cnn_models': [
        'vgg16', 'resnet50', 'densenet121', 'xception'
    ],
    'vit_models': [
        'mobilevit', 'maxvit', 'mvitv2', 'deit3'
    ]
}

# Models Status and Priority
MODELS_STATUS = {
    # CNN Models
    'vgg16': {'enabled': True, 'priority': 1, 'trained': True, 'accuracy': 0.84},
    'resnet50': {'enabled': True, 'priority': 2, 'trained': False, 'accuracy': 0.0},
    'densenet121': {'enabled': True, 'priority': 3, 'trained': False, 'accuracy': 0.0},
    'xception': {'enabled': True, 'priority': 4, 'trained': False, 'accuracy': 0.0},
    
    # Vision Transformer Models
    'mobilevit': {'enabled': True, 'priority': 1, 'trained': False, 'accuracy': 0.0},
    'maxvit': {'enabled': True, 'priority': 2, 'trained': False, 'accuracy': 0.0},
    'mvitv2': {'enabled': True, 'priority': 3, 'trained': False, 'accuracy': 0.0},
    'deit3': {'enabled': True, 'priority': 4, 'trained': False, 'accuracy': 0.0}
}

# Comprehensive Random Search Parameters for All Models
RANDOM_SEARCH_PARAMS = {
    # CNN Models Parameters
    'vgg16': {
        'learning_rate': [5e-5, 1e-4, 2e-4],  # Modern lower learning rates
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.4, 0.5, 0.6, 0.7],  # Higher dropout for regularization
        'optimizer': ['adamw'],  # Modern optimizer with weight decay
        'weight_decay': [1e-3, 5e-3, 1e-2],  # Add weight decay for regularization
        'momentum': [0.9]
    },
    'resnet50': {
        'learning_rate': [5e-5, 1e-4, 1e-4],  # Lower learning rates
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.4, 0.5, 0.6, 0.7],  # Higher dropout rates
        'optimizer': ['adamw'],  # Focus on AdamW for better regularization
        'weight_decay': [5e-3, 1e-2, 2e-2],  # Much higher weight decay
        'momentum': [0.9]
    },
    'densenet121': {
        'learning_rate': [5e-5, 1e-4, 1e-4],  # Lower learning rates
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.4, 0.5, 0.6, 0.7],  # Higher dropout rates
        'optimizer': ['adamw'],  # Focus on AdamW
        'weight_decay': [5e-3, 1e-2, 2e-2],  # Much higher weight decay
        'growth_rate': [16, 24, 32]  # Slightly constrained growth
    },
    'xception': {
        'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.3, 0.4, 0.5],
        'optimizer': ['adam', 'adamw'],
        'weight_decay': [1e-5, 1e-4],
        'label_smoothing': [0.0, 0.1, 0.2]
    },
    
    # Vision Transformer Models Parameters
    'mobilevit': {
        'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4, 5e-4],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'mlp_dim': [256, 384, 512, 640],
        'num_heads': [2, 4, 6, 8],
        'patch_size': [2, 4]
    },
    'maxvit': {
        'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.1, 0.2, 0.3],
        'mlp_dim': [256, 384, 512, 640],
        'num_heads': [4, 8, 16]
    },
    'mvitv2': {
        'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.1, 0.2, 0.3],
        'model_size': ['tiny', 'small', 'base']
    },
    'deit3': {
        'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.1, 0.2, 0.3],
        'num_heads': [6, 8, 12],
        'mlp_ratio': [3, 4, 6],
        'patch_size': [16, 32]
    }
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'cnn_ensemble': {
        'voting_method': 'soft',
        'min_models': 2,
        'weight_by_accuracy': True,
        'threshold_accuracy': 0.75  # Minimum accuracy to include in ensemble
    },
    'vit_ensemble': {
        'voting_method': 'soft',
        'min_models': 2,
        'weight_by_accuracy': True,
        'threshold_accuracy': 0.70  # Minimum accuracy to include in ensemble
    },
    'final_ensemble': {
        'voting_method': 'soft',
        'cnn_weight': 0.5,
        'vit_weight': 0.5,
        'adaptive_weighting': True  # Adjust weights based on performance
    }
}

# Training Pipeline Configuration
TRAINING_CONFIG = {
    'save_checkpoints': True,
    'checkpoint_frequency': 5,  # Save every 5 epochs
    'log_frequency': 1,  # Log every epoch
    'validate_frequency': 1,  # Validate every epoch
    'save_best_only': True,
    'monitor_metric': 'val_accuracy',
    'mode': 'max'
}

# Paths Configuration
PATHS = {
    'models': 'models',
    'results': 'results',
    'logs': 'logs',
    'checkpoints': 'checkpoints',
    'ensemble': 'results/ensemble'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.60,  # Minimum acceptable accuracy
    'excellent_accuracy': 0.85,  # Excellent performance threshold
    'early_stop_improvement': 0.001  # Minimum improvement for early stopping
}

# Model Categories Configuration
MODELS_CONFIG = {
    'cnn': ['vgg16', 'resnet50', 'densenet121', 'xception'],
    'vit': ['mobilevit', 'maxvit', 'mvitv2', 'deit3']
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'voting_method': 'soft',  # 'soft' or 'hard'
    'top_k_models': 3,  # Number of best models to include in ensemble
    'weight_method': 'accuracy'  # How to weight models: 'equal' or 'accuracy'
}