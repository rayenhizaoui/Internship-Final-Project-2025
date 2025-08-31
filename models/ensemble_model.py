"""
Comprehensive Multi-Model Training and Ensemble System
Handles 8 models: VGG16, ResNet50, DenseNet121, Xception, MobileViT, MaxViT, MViTv2, DeiT3
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
import logging

# Import all model factories
from models.vgg16_model import create_vgg16_model
from models.resnet50_model import create_resnet50_model
from models.densenet121_model import create_densenet121_model
from models.xception_model import create_xception_model
from models.mobilevit_model import create_maxvit_model as create_mobilevit_model
from models.maxvit_model import create_mobilevit_model as create_maxvit_model
from models.mvitv2_model import create_mvitv2_model
from models.deit3_model import create_deit3_model


class ModelFactory:
    """Factory class to create all supported models."""
    
    MODEL_CREATORS = {
        'vgg16': create_vgg16_model,
        'resnet50': create_resnet50_model,
        'densenet121': create_densenet121_model,
        'xception': create_xception_model,
        'mobilevit': create_mobilevit_model,
        'maxvit': create_maxvit_model,
        'mvitv2': create_mvitv2_model,
        'deit3': create_deit3_model
    }
    
    @classmethod
    def create_model(cls, model_name: str, num_classes: int = 3, **kwargs):
        """Create a model by name."""
        if model_name not in cls.MODEL_CREATORS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODEL_CREATORS.keys())}")
        
        return cls.MODEL_CREATORS[model_name](num_classes=num_classes, **kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls.MODEL_CREATORS.keys())


class EnsembleModel(nn.Module):
    """Ensemble model that combines multiple trained models."""
    
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None,
                 voting_method: str = 'soft'):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.voting_method = voting_method
        
        # Initialize weights
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models.keys()}
        
        self.weights = nn.Parameter(torch.tensor([weights[name] for name in models.keys()]))
        self.model_names = list(models.keys())
        
    def forward(self, x):
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models.values():
            model.eval()
            with torch.no_grad():
                output = model(x)
                if self.voting_method == 'soft':
                    output = torch.softmax(output, dim=1)
                outputs.append(output)
        
        # Stack outputs and apply weights
        outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, num_classes)
        
        if self.voting_method == 'soft':
            # Weighted average of softmax outputs
            weighted_output = torch.sum(outputs * self.weights.view(-1, 1, 1), dim=0)
        else:  # hard voting
            # Take argmax then majority vote
            predictions = torch.argmax(outputs, dim=2)
            weighted_output = torch.mode(predictions, dim=0)[0]
        
        return weighted_output


class MultiModelTrainer:
    """Trainer for managing multiple models with ensemble capabilities."""
    
    def __init__(self, config, data_loaders):
        self.config = config
        self.data_loaders = data_loaders
        self.models = {}
        self.trained_models = {}
        self.model_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_all_models(self) -> Dict[str, nn.Module]:
        """Create all models according to configuration."""
        models = {}
        
        for category in ['cnn_models', 'vit_models']:
            for model_name in self.config.MODEL_CATEGORIES[category]:
                if self.config.MODELS_STATUS[model_name]['enabled']:
                    try:
                        model = ModelFactory.create_model(
                            model_name, 
                            num_classes=self.config.NUM_CLASSES
                        )
                        models[model_name] = model
                        self.logger.info(f"Created {model_name} model successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to create {model_name}: {e}")
        
        self.models = models
        return models
    
    def load_trained_model(self, model_name: str, model_path: str) -> Optional[nn.Module]:
        """Load a trained model from checkpoint."""
        if not os.path.exists(model_path):
            self.logger.warning(f"Model checkpoint not found: {model_path}")
            return None
        
        try:
            model = self.models.get(model_name)
            if model is None:
                model = ModelFactory.create_model(model_name, num_classes=self.config.NUM_CLASSES)
            
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Store metrics if available
            if 'metrics' in checkpoint:
                self.model_metrics[model_name] = checkpoint['metrics']
            
            self.trained_models[model_name] = model
            self.logger.info(f"Loaded trained {model_name} model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} from {model_path}: {e}")
            return None
    
    def create_cnn_ensemble(self, min_accuracy: float = 0.75) -> Optional[EnsembleModel]:
        """Create ensemble of CNN models."""
        cnn_models = {}
        weights = {}
        
        for model_name in self.config.MODEL_CATEGORIES['cnn_models']:
            if model_name in self.trained_models:
                accuracy = self.model_metrics.get(model_name, {}).get('val_accuracy', 0.0)
                if accuracy >= min_accuracy:
                    cnn_models[model_name] = self.trained_models[model_name]
                    weights[model_name] = accuracy  # Weight by accuracy
        
        if len(cnn_models) < 2:
            self.logger.warning(f"Not enough CNN models for ensemble (need 2, have {len(cnn_models)})")
            return None
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        ensemble = EnsembleModel(cnn_models, weights, voting_method='soft')
        self.logger.info(f"Created CNN ensemble with {len(cnn_models)} models: {list(cnn_models.keys())}")
        return ensemble
    
    def create_vit_ensemble(self, min_accuracy: float = 0.70) -> Optional[EnsembleModel]:
        """Create ensemble of Vision Transformer models."""
        vit_models = {}
        weights = {}
        
        for model_name in self.config.MODEL_CATEGORIES['vit_models']:
            if model_name in self.trained_models:
                accuracy = self.model_metrics.get(model_name, {}).get('val_accuracy', 0.0)
                if accuracy >= min_accuracy:
                    vit_models[model_name] = self.trained_models[model_name]
                    weights[model_name] = accuracy  # Weight by accuracy
        
        if len(vit_models) < 2:
            self.logger.warning(f"Not enough ViT models for ensemble (need 2, have {len(vit_models)})")
            return None
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        ensemble = EnsembleModel(vit_models, weights, voting_method='soft')
        self.logger.info(f"Created ViT ensemble with {len(vit_models)} models: {list(vit_models.keys())}")
        return ensemble
    
    def create_final_ensemble(self, cnn_ensemble: EnsembleModel, vit_ensemble: EnsembleModel,
                            cnn_weight: float = 0.5) -> EnsembleModel:
        """Create final ensemble combining CNN and ViT ensembles."""
        final_models = {
            'cnn_ensemble': cnn_ensemble,
            'vit_ensemble': vit_ensemble
        }
        
        weights = {
            'cnn_ensemble': cnn_weight,
            'vit_ensemble': 1.0 - cnn_weight
        }
        
        ensemble = EnsembleModel(final_models, weights, voting_method='soft')
        self.logger.info(f"Created final ensemble with CNN weight: {cnn_weight}, ViT weight: {1.0 - cnn_weight}")
        return ensemble
    
    def evaluate_ensemble(self, ensemble: EnsembleModel, data_loader) -> Dict[str, float]:
        """Evaluate ensemble model performance."""
        ensemble.eval()
        correct = 0
        total = 0
        device = next(ensemble.parameters()).device
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs)
                
                if len(outputs.shape) == 1:  # Hard voting
                    predicted = outputs
                else:  # Soft voting
                    predicted = torch.argmax(outputs, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return {'accuracy': accuracy}
    
    def save_ensemble(self, ensemble: EnsembleModel, save_path: str, metrics: Dict = None):
        """Save ensemble model and its configuration."""
        ensemble_state = {
            'model_state_dict': ensemble.state_dict(),
            'model_names': ensemble.model_names,
            'voting_method': ensemble.voting_method,
            'metrics': metrics or {}
        }
        
        torch.save(ensemble_state, save_path)
        self.logger.info(f"Saved ensemble to {save_path}")
    
    def load_ensemble(self, load_path: str) -> Optional[EnsembleModel]:
        """Load ensemble model from checkpoint."""
        if not os.path.exists(load_path):
            self.logger.warning(f"Ensemble checkpoint not found: {load_path}")
            return None
        
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            # Recreate models
            models = {}
            for model_name in checkpoint['model_names']:
                model = ModelFactory.create_model(model_name, num_classes=self.config.NUM_CLASSES)
                models[model_name] = model
            
            ensemble = EnsembleModel(models, voting_method=checkpoint['voting_method'])
            ensemble.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"Loaded ensemble from {load_path}")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble from {load_path}: {e}")
            return None
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models and their status."""
        summary = {
            'total_models': len(self.models),
            'trained_models': len(self.trained_models),
            'model_details': {}
        }
        
        for model_name, model in self.models.items():
            params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            summary['model_details'][model_name] = {
                'parameters': params,
                'trainable_parameters': trainable_params,
                'is_trained': model_name in self.trained_models,
                'metrics': self.model_metrics.get(model_name, {})
            }
        
        return summary


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")
    
    # Test creating models
    for model_name in ModelFactory.get_available_models():
        try:
            model = ModelFactory.create_model(model_name, num_classes=3)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {model_name}: {params:,} parameters")
        except Exception as e:
            print(f"✗ {model_name}: {e}")
    
    print(f"\nTotal available models: {len(ModelFactory.get_available_models())}")
    print("Model Factory test completed!")
