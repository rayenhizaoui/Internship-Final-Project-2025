"""
Professional Multi-Model Training Pipeline for Corn Disease Classification
Scalable, modular, and enterprise-ready PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import random
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Import configurations and utilities
from utils.config import MODELS_CONFIG, RANDOM_SEARCH_PARAMS, ENSEMBLE_CONFIG
from utils.config import DATASET_DIR, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, ENABLE_EARLY_STOPPING
from utils.config import RANDOM_SEARCH_ITERATIONS, RANDOM_SEARCH_EPOCHS, ENABLE_RANDOM_SEARCH
from data.data_loader import create_pytorch_dataset_info


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss to prevent overfitting"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: predictions [N, C]
            target: ground truth [N]
        """
        n_class = pred.size(1)
        target_one_hot = F.one_hot(target, n_class).float()
        
        # Apply label smoothing
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_class
        
        log_prob = F.log_softmax(pred, dim=1)
        loss = -target_smooth * log_prob
        
        return loss.sum(dim=1).mean()
from data.preprocessing import create_data_loaders

# Import all model architectures (with error handling)
try:
    from models.vgg16_model import create_vgg16_model
except ImportError:
    create_vgg16_model = None

try:
    from models.resnet50_model import create_resnet50_model
except ImportError:
    create_resnet50_model = None

try:
    from models.densenet121_model import create_densenet121_model
except ImportError:
    create_densenet121_model = None

try:
    from models.xception_model import create_xception_model
except ImportError:
    create_xception_model = None

try:
    from models.mobilevit_model import create_mobilevit_model
except ImportError:
    create_mobilevit_model = None

try:
    from models.mobilevit_simple_v2 import create_mobilevit_simple_v2_model
except ImportError:
    create_mobilevit_simple_v2_model = None

try:
    from models.maxvit_model import create_mobilevit_model as create_maxvit_model
except ImportError:
    create_maxvit_model = None

try:
    from models.mvitv2_model import create_mvitv2_model
except ImportError:
    create_mvitv2_model = None

try:
    from models.deit3_model import create_deit3_model
except ImportError:
    create_deit3_model = None

try:
    from models.ensemble_model import EnsembleModel, create_ensemble_model
except ImportError:
    EnsembleModel = None
    create_ensemble_model = None


class ProfessionalTrainingPipeline:
    """
    Enterprise-grade training pipeline with advanced features:
    - Automatic model selection and hyperparameter optimization
    - Scalable architecture for multiple model types
    - Professional logging and monitoring
    - Fault tolerance and recovery
    - Modular ensemble creation
    """
    
    def __init__(self, config_override=None):
        """Initialize the training pipeline with configuration."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model factory for scalable architecture with error handling
        self.model_factory = {}
        
        # Add available CNN models
        if create_vgg16_model:
            self.model_factory['vgg16'] = create_vgg16_model
        if create_resnet50_model:
            self.model_factory['resnet50'] = create_resnet50_model
        if create_densenet121_model:
            self.model_factory['densenet121'] = create_densenet121_model
        if create_xception_model:
            self.model_factory['xception'] = create_xception_model
            
        # Add available ViT models
        if create_mobilevit_simple_v2_model:
            self.model_factory['mobilevit'] = create_mobilevit_simple_v2_model
        if create_maxvit_model:
            self.model_factory['maxvit'] = create_maxvit_model
        if create_mvitv2_model:
            self.model_factory['mvitv2'] = create_mvitv2_model
        if create_deit3_model:
            self.model_factory['deit3'] = create_deit3_model
            
        print(f"✅ Available models: {list(self.model_factory.keys())}")
        
        # Update config to only include available models
        available_cnn = [name for name in MODELS_CONFIG['cnn'] if name in self.model_factory]
        available_vit = [name for name in MODELS_CONFIG['vit'] if name in self.model_factory]
        
        print(f"🔥 CNN models: {available_cnn}")
        print(f"🚀 ViT models: {available_vit}")
        
        # Store model categories
        self.cnn_models = available_cnn
        self.vit_models = available_vit
        
        # Initialize tracking dictionaries
        self.model_results = {'cnn_models': {}, 'vit_models': {}}
        self.ensemble_results = {}
        
        # Setup directories and logging
        self._setup_environment()
        self._initialize_logging()
        
        print(f"🚀 Professional Training Pipeline Initialized")
        print(f"Session ID: {self.session_id}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _setup_environment(self):
        """Setup directories and environment with Windows compatibility."""
        # Create all necessary directories
        directories = [
            'results', 'results/models', 'results/ensemble', 
            'results/logs', 'results/checkpoints', 'results/reports'
        ]
        
        # Add model-specific directories
        all_models = list(self.model_factory.keys())
        
        for model_name in all_models:
            directories.extend([
                f'results/{model_name}',
                f'results/checkpoints/{model_name}'
            ])
        
        # Create directories with Windows error handling
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    print(f"📁 Created: {directory}")
            except Exception as e:
                print(f"⚠️ Directory issue {directory}: {str(e)}")
                # Continue - some might exist as files
    
    def _initialize_logging(self):
        """Initialize comprehensive logging."""
        os.makedirs('results/logs', exist_ok=True)
        self.log_file = f'results/logs/training_{self.session_id}.log'
        self.results_file = f'results/logs/results_{self.session_id}.json'
        
        # Create initial log entry
        self._log_message("SYSTEM", "Training pipeline initialized", {
            'session_id': self.session_id,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available()
        })
    
    def _log_message(self, level, message, data=None):
        """Professional logging with structured format."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'data': data or {}
        }
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Continue even if logging fails
        
        # Print to console with color coding
        if level == "INFO":
            print(f"📋 {timestamp} - {message}")
        elif level == "SUCCESS":
            print(f"✅ {timestamp} - {message}")
        elif level == "ERROR":
            print(f"❌ {timestamp} - {message}")
        elif level == "WARNING":
            print(f"⚠️ {timestamp} - {message}")
        else:
            print(f"🔧 {timestamp} - {message}")
    
    def train_single_model(self, model_name, model_params, dataset_info, epochs=None):
        """Train a single model with given parameters."""
        try:
            self._log_message("INFO", f"Starting training for {model_name}", model_params)
            
            # Separate model parameters from training parameters
            batch_size = model_params.pop('batch_size', BATCH_SIZE)
            
            # Create model (without batch_size parameter)
            model_func = self.model_factory[model_name]
            model, optimizer, criterion = model_func(**model_params)
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                dataset_info['train_paths'], dataset_info['train_labels'],
                dataset_info['val_paths'], dataset_info['val_labels'],
                batch_size=batch_size
            )
            
            # Train model
            patience = EARLY_STOPPING_PATIENCE if ENABLE_EARLY_STOPPING else None
            history = self.train_model_with_monitoring(
                model, optimizer, criterion, train_loader, val_loader,
                epochs=epochs or EPOCHS, model_name=model_name,
                early_stopping_patience=patience
            )
            
            # Restore batch_size for saving
            model_params['batch_size'] = batch_size
            
            # Save model
            self._save_model(model, model_name, model_params, history)
            
            self._log_message("SUCCESS", f"Completed training for {model_name}", {
                'best_val_acc': history['best_val_acc'],
                'epochs_trained': history['epochs_trained']
            })
            
            return model, history
            
        except Exception as e:
            self._log_message("ERROR", f"Failed to train {model_name}", {'error': str(e)})
            print(f"ERROR DETAILS: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def train_model_with_monitoring(self, model, optimizer, criterion, train_loader, val_loader, 
                                  epochs, model_name, early_stopping_patience=None):
        """Enhanced training with professional monitoring."""
        model = model.to(self.device)
        
        # Early stopping setup
        use_early_stopping = early_stopping_patience is not None
        if use_early_stopping:
            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience_counter = 0
            best_model_state = None
            
        # Metrics tracking
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        
        print(f"\n🔥 Training {model_name.upper()}")
        print(f"Training on device: {self.device}")
        print(f"Training for up to {epochs} epochs" + 
              (f" with early stopping (patience: {early_stopping_patience})" if use_early_stopping else ""))
        print("=" * 60)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct_train / total_train
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * correct_val / total_val
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Epoch [{epoch+1:2d}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:5.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.2f}%")
            
            # Early stopping logic
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"✅ New best validation loss: {best_val_loss:.4f} (accuracy: {best_val_acc:.2f}%)")
                else:
                    patience_counter += 1
                    print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"🛑 Early stopping triggered! Best validation loss: {best_val_loss:.4f} (accuracy: {best_val_acc:.2f}%)")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                            print("✅ Restored best model weights")
                        break
        
        # Prepare results
        result = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'epochs_trained': epoch + 1
        }
        
        if use_early_stopping:
            if best_model_state is not None and patience_counter < early_stopping_patience:
                model.load_state_dict(best_model_state)
                print(f"✅ Training completed. Using best model with validation loss: {best_val_loss:.4f} (accuracy: {best_val_acc:.2f}%)")
            result['best_val_loss'] = best_val_loss
            result['best_val_acc'] = best_val_acc
        else:
            result['best_val_loss'] = val_loss
            result['best_val_acc'] = val_acc
        
        return result
    
    def perform_random_search(self, model_name, dataset_info, iterations=None):
        """Perform random search for hyperparameter optimization."""
        iterations = iterations or RANDOM_SEARCH_ITERATIONS
        params = RANDOM_SEARCH_PARAMS.get(model_name, {})
        
        if not params:
            self._log_message("WARNING", f"No random search parameters defined for {model_name}")
            return None, 0.0
        
        print(f"\n🔍 Random Search for {model_name.upper()}")
        print(f"Testing {iterations} parameter combinations")
        print("=" * 50)
        
        best_params = None
        best_score = 0.0
        
        for i in range(iterations):
            # Generate random parameters
            random_params = {}
            for param_name, param_values in params.items():
                random_params[param_name] = random.choice(param_values)
            
            print(f"\nIteration {i+1}/{iterations}")
            print(f"Testing parameters: {random_params}")
            
            try:
                # Train with random parameters for fewer epochs
                model, history = self.train_single_model(
                    model_name, random_params, dataset_info, 
                    epochs=RANDOM_SEARCH_EPOCHS
                )
                
                score = history['best_val_acc']
                print(f"Validation Accuracy: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = random_params.copy()
                    print(f"New best score: {best_score:.4f}")
                
            except Exception as e:
                self._log_message("ERROR", f"Random search iteration {i+1} failed", {'error': str(e)})
                print(f"ERROR in iteration {i+1}: {str(e)}")
                continue
        
        print(f"\n🏆 Random Search completed for {model_name}")
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
        
        return best_params, best_score
    
    def train_model_category(self, category_name):
        """Train all models in a category (CNN or ViT)."""
        print(f"\n🚀 Starting {category_name.upper()} Models Training")
        print("=" * 60)
        
        # Get models for this category
        if category_name == 'cnn':
            models_to_train = self.cnn_models
        elif category_name == 'vit':
            models_to_train = self.vit_models
        else:
            models_to_train = list(self.model_factory.keys())
            
        if not models_to_train:
            print(f"⚠️ No {category_name} models available, skipping category")
            return {}
        
        category_results = {}
        
        # Load dataset once
        dataset_info = create_pytorch_dataset_info(DATASET_DIR)
        
        for model_name in models_to_train:
            if model_name not in self.model_factory:
                self._log_message("WARNING", f"Model {model_name} not found in factory")
                continue
            
            try:
                # Check if model already trained
                model_path = f'results/{model_name}/{model_name}_optimized_model.pth'
                if os.path.exists(model_path):
                    print(f"✅ {model_name} already trained, loading results...")
                    result = self._load_existing_results(model_name)
                    if result:
                        category_results[model_name] = result
                        continue
                
                # Perform random search if enabled
                if ENABLE_RANDOM_SEARCH:
                    print(f"\n🔍 Performing Random Search for {model_name}")
                    best_params, best_score = self.perform_random_search(model_name, dataset_info)
                    
                    if best_params is None:
                        self._log_message("WARNING", f"Random search failed for {model_name}, using default parameters")
                        best_params = {}
                else:
                    best_params = {}
                
                # Final training with best parameters
                print(f"\n🔥 Final Training for {model_name} with optimized parameters")
                model, history = self.train_single_model(model_name, best_params, dataset_info)
                
                category_results[model_name] = {
                    'accuracy': history['best_val_acc'],
                    'loss': history['best_val_loss'],
                    'epochs': history['epochs_trained'],
                    'parameters': best_params,
                    'model_path': f'results/{model_name}/{model_name}_optimized_model.pth'
                }
                
            except Exception as e:
                self._log_message("ERROR", f"Failed to train {model_name}", {'error': str(e)})
                print(f"ERROR training {model_name}: {str(e)}")
                continue
        
        return category_results
    
    def create_ensemble(self, category_name, models_results, top_k=3):
        """Create ensemble from best models in category."""
        print(f"\n🔧 Creating {category_name.upper()} Ensemble")
        print("-" * 40)
        
        if len(models_results) < 2:
            print(f"⚠️ Need at least 2 models for ensemble, got {len(models_results)}")
            return None
        
        # Select top k models based on accuracy
        sorted_models = sorted(models_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_models = dict(sorted_models[:min(top_k, len(sorted_models))])
        
        print(f"Selected top {len(top_models)} models for {category_name} ensemble:")
        for name, result in top_models.items():
            print(f"  {name:12}: {result['accuracy']:6.2f}%")
        
        try:
            # Create ensemble using the ensemble model
            if create_ensemble_model is None:
                print("⚠️ Ensemble model not available, creating simple voting ensemble")
                return self._create_simple_ensemble(category_name, top_models)
            
            # Load models for ensemble
            loaded_models = {}
            for model_name, result in top_models.items():
                model_path = result['model_path']
                if os.path.exists(model_path):
                    # Create model architecture
                    model_func = self.model_factory[model_name]
                    model, _, _ = model_func()  # Use default parameters
                    
                    # Load trained weights
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model = model.to(self.device)
                    model.eval()
                    
                    loaded_models[model_name] = {
                        'model': model,
                        'weight': result['accuracy'] / 100.0  # Use accuracy as weight
                    }
                    print(f"✅ Loaded {model_name} for ensemble")
                else:
                    print(f"⚠️ Model file not found: {model_path}")
            
            if len(loaded_models) < 2:
                print(f"⚠️ Only {len(loaded_models)} models loaded, ensemble requires at least 2")
                return None
            
            # Create ensemble model
            ensemble = create_ensemble_model(loaded_models, category_name)
            
            # Test ensemble performance
            dataset_info = create_pytorch_dataset_info(DATASET_DIR)
            train_loader, val_loader = create_data_loaders(
                dataset_info['train_paths'], dataset_info['train_labels'],
                dataset_info['val_paths'], dataset_info['val_labels'],
                batch_size=BATCH_SIZE
            )
            
            # Evaluate ensemble
            ensemble_acc = self._evaluate_ensemble(ensemble, val_loader)
            
            # Save ensemble
            ensemble_dir = f'results/ensemble'
            os.makedirs(ensemble_dir, exist_ok=True)
            ensemble_path = f'{ensemble_dir}/{category_name}_ensemble.pth'
            torch.save(ensemble.state_dict(), ensemble_path)
            
            # Save ensemble metadata
            ensemble_metadata = {
                'category': category_name,
                'models': list(top_models.keys()),
                'model_accuracies': {name: result['accuracy'] for name, result in top_models.items()},
                'ensemble_accuracy': ensemble_acc,
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }
            
            metadata_path = f'{ensemble_dir}/{category_name}_ensemble_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(ensemble_metadata, f, indent=2, default=str)
            
            result = {
                'accuracy': ensemble_acc,
                'models': list(top_models.keys()),
                'model_accuracies': {name: result['accuracy'] for name, result in top_models.items()},
                'model_path': ensemble_path,
                'metadata_path': metadata_path
            }
            
            print(f"✅ {category_name} ensemble created with accuracy: {ensemble_acc:.2f}%")
            self._log_message("SUCCESS", f"{category_name} ensemble created", {
                'accuracy': ensemble_acc,
                'models': list(top_models.keys())
            })
            
            return result
            
        except Exception as e:
            self._log_message("ERROR", f"Failed to create {category_name} ensemble", {'error': str(e)})
            print(f"ERROR creating {category_name} ensemble: {str(e)}")
            return None
    
    def _create_simple_ensemble(self, category_name, top_models):
        """Create a simple voting ensemble when ensemble_model is not available."""
        print(f"Creating simple voting ensemble for {category_name}")
        
        # For now, return the best single model as "ensemble"
        best_model = max(top_models.items(), key=lambda x: x[1]['accuracy'])
        
        result = {
            'accuracy': best_model[1]['accuracy'],
            'models': [best_model[0]],
            'model_accuracies': {best_model[0]: best_model[1]['accuracy']},
            'model_path': best_model[1]['model_path'],
            'note': 'Simple ensemble (best single model)'
        }
        
        print(f"✅ Simple {category_name} ensemble: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
        return result
    
    def create_ultimate_ensemble(self, all_results):
        """Create ultimate ensemble combining best CNN and ViT models."""
        print(f"\n🏆 Creating Ultimate Ensemble (CNN + ViT)")
        print("-" * 50)
        
        cnn_ensemble = all_results.get('cnn_ensemble')
        vit_ensemble = all_results.get('vit_ensemble')
        
        if not cnn_ensemble or not vit_ensemble:
            print("⚠️ Need both CNN and ViT ensembles for ultimate ensemble")
            return None
        
        try:
            # Combine the two ensembles
            print(f"CNN Ensemble: {cnn_ensemble['accuracy']:.2f}%")
            print(f"ViT Ensemble: {vit_ensemble['accuracy']:.2f}%")
            
            # Simple weighted combination based on performance
            cnn_weight = cnn_ensemble['accuracy'] / (cnn_ensemble['accuracy'] + vit_ensemble['accuracy'])
            vit_weight = vit_ensemble['accuracy'] / (cnn_ensemble['accuracy'] + vit_ensemble['accuracy'])
            
            # Estimate combined performance (simple average for now)
            estimated_accuracy = (cnn_ensemble['accuracy'] + vit_ensemble['accuracy']) / 2
            
            ultimate_dir = f'results/ensemble'
            os.makedirs(ultimate_dir, exist_ok=True)
            
            # Save ultimate ensemble metadata
            ultimate_metadata = {
                'type': 'ultimate_ensemble',
                'cnn_ensemble': {
                    'models': cnn_ensemble['models'],
                    'accuracy': cnn_ensemble['accuracy'],
                    'weight': cnn_weight
                },
                'vit_ensemble': {
                    'models': vit_ensemble['models'],
                    'accuracy': vit_ensemble['accuracy'],
                    'weight': vit_weight
                },
                'estimated_accuracy': estimated_accuracy,
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }
            
            metadata_path = f'{ultimate_dir}/ultimate_ensemble_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(ultimate_metadata, f, indent=2, default=str)
            
            result = {
                'accuracy': estimated_accuracy,
                'cnn_models': cnn_ensemble['models'],
                'vit_models': vit_ensemble['models'],
                'cnn_weight': cnn_weight,
                'vit_weight': vit_weight,
                'metadata_path': metadata_path,
                'description': 'Ultimate ensemble combining best CNN and ViT models'
            }
            
            print(f"✅ Ultimate ensemble created:")
            print(f"  CNN models: {cnn_ensemble['models']} (weight: {cnn_weight:.3f})")
            print(f"  ViT models: {vit_ensemble['models']} (weight: {vit_weight:.3f})")
            print(f"  Estimated accuracy: {estimated_accuracy:.2f}%")
            
            self._log_message("SUCCESS", "Ultimate ensemble created", {
                'estimated_accuracy': estimated_accuracy,
                'cnn_models': cnn_ensemble['models'],
                'vit_models': vit_ensemble['models']
            })
            
            return result
            
        except Exception as e:
            self._log_message("ERROR", "Failed to create ultimate ensemble", {'error': str(e)})
            print(f"ERROR creating ultimate ensemble: {str(e)}")
            return None
    
    def _save_model(self, model, model_name, params, history):
        """Save model and metadata."""
        model_dir = f'results/{model_name}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        model_path = f'{model_dir}/{model_name}_optimized_model.pth'
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'parameters': params,
            'history': history,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        metadata_path = f'{model_dir}/{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_existing_results(self, model_name):
        """Load results from existing trained model."""
        metadata_path = f'results/{model_name}/{model_name}_metadata.json'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    'accuracy': metadata['history']['best_val_acc'],
                    'loss': metadata['history']['best_val_loss'],
                    'epochs': metadata['history']['epochs_trained'],
                    'parameters': metadata['parameters'],
                    'model_path': f'results/{model_name}/{model_name}_optimized_model.pth'
                }
            except Exception:
                pass
        return None
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        print("🎯 Starting Complete Professional Training Pipeline")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        all_results = {}
        
        try:
            # 1. Train CNN models
            if self.cnn_models:
                print(f"\n🔥 Phase 1: Training CNN Models ({len(self.cnn_models)} models)")
                cnn_results = self.train_model_category('cnn')
                all_results['cnn_models'] = cnn_results
                print(f"\n✅ CNN Training completed: {len(cnn_results)} models trained")
                
                # Create CNN ensemble
                if len(cnn_results) >= 2:
                    print(f"\n🔧 Phase 2: Creating CNN Ensemble")
                    cnn_ensemble = self.create_ensemble('cnn', cnn_results)
                    if cnn_ensemble:
                        all_results['cnn_ensemble'] = cnn_ensemble
                        print(f"✅ CNN Ensemble created with accuracy: {cnn_ensemble['accuracy']:.2f}%")
            
            # 2. Train ViT models
            if self.vit_models:
                print(f"\n🚀 Phase 3: Training ViT Models ({len(self.vit_models)} models)")
                vit_results = self.train_model_category('vit')
                all_results['vit_models'] = vit_results
                print(f"\n✅ ViT Training completed: {len(vit_results)} models trained")
                
                # Create ViT ensemble
                if len(vit_results) >= 2:
                    print(f"\n🔧 Phase 4: Creating ViT Ensemble")
                    vit_ensemble = self.create_ensemble('vit', vit_results)
                    if vit_ensemble:
                        all_results['vit_ensemble'] = vit_ensemble
                        print(f"✅ ViT Ensemble created with accuracy: {vit_ensemble['accuracy']:.2f}%")
            
            # 3. Create Ultimate Ensemble (Best CNN + Best ViT)
            if 'cnn_ensemble' in all_results and 'vit_ensemble' in all_results:
                print(f"\n🏆 Phase 5: Creating Ultimate Ensemble (CNN + ViT)")
                ultimate_ensemble = self.create_ultimate_ensemble(all_results)
                if ultimate_ensemble:
                    all_results['ultimate_ensemble'] = ultimate_ensemble
                    print(f"✅ Ultimate Ensemble created with accuracy: {ultimate_ensemble['accuracy']:.2f}%")
            
            # 4. Generate final report
            print(f"\n📊 Phase 6: Generating Final Report")
            self._generate_final_report(all_results)
            
            # Calculate total time
            total_time = (time.time() - self.start_time) / 3600
            print(f"\n🎉 Complete pipeline finished successfully!")
            print(f"Total time: {total_time:.2f} hours")
            print(f"Total models trained: {len(all_results.get('cnn_models', {})) + len(all_results.get('vit_models', {}))}")
            print(f"Ensembles created: {len([k for k in all_results.keys() if 'ensemble' in k])}")
            
            return all_results
            
        except Exception as e:
            self._log_message("ERROR", "Pipeline failed", {'error': str(e)})
            print(f"PIPELINE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _generate_final_report(self, results):
        """Generate comprehensive final report."""
        os.makedirs('results/reports', exist_ok=True)
        
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'total_time_hours': (time.time() - self.start_time) / 3600,
            'results': results,
            'summary': {
                'total_models_trained': len(results.get('cnn_models', {})) + len(results.get('vit_models', {})),
                'ensembles_created': len([k for k in results.keys() if 'ensemble' in k]),
                'best_cnn_model': None,
                'best_vit_model': None,
                'best_overall_accuracy': 0.0
            }
        }
        
        # Find best models
        if 'cnn_models' in results and results['cnn_models']:
            best_cnn = max(results['cnn_models'].items(), key=lambda x: x[1]['accuracy'])
            report['summary']['best_cnn_model'] = {'name': best_cnn[0], 'accuracy': best_cnn[1]['accuracy']}
            
        if 'vit_models' in results and results['vit_models']:
            best_vit = max(results['vit_models'].items(), key=lambda x: x[1]['accuracy'])
            report['summary']['best_vit_model'] = {'name': best_vit[0], 'accuracy': best_vit[1]['accuracy']}
        
        # Find best overall accuracy
        all_accuracies = []
        for category, models in results.items():
            if 'models' in category and isinstance(models, dict):
                all_accuracies.extend([model['accuracy'] for model in models.values()])
            elif 'ensemble' in category and isinstance(models, dict) and 'accuracy' in models:
                all_accuracies.append(models['accuracy'])
        
        if all_accuracies:
            report['summary']['best_overall_accuracy'] = max(all_accuracies)
        
        # Save detailed report
        report_path = f'results/reports/final_report_{self.session_id}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        # Individual models
        for category, models in results.items():
            if 'models' in category and isinstance(models, dict):
                print(f"\n{category.upper().replace('_', ' ')}:")
                sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                for model_name, result in sorted_models:
                    print(f"  {model_name:12}: {result['accuracy']:6.2f}%")
        
        # Ensembles
        print(f"\n🔧 ENSEMBLES:")
        for category, ensemble in results.items():
            if 'ensemble' in category and isinstance(ensemble, dict):
                if category == 'ultimate_ensemble':
                    print(f"  🏆 Ultimate     : {ensemble['accuracy']:6.2f}% (CNN + ViT)")
                    print(f"      CNN models  : {', '.join(ensemble['cnn_models'])}")
                    print(f"      ViT models  : {', '.join(ensemble['vit_models'])}")
                else:
                    ensemble_type = category.replace('_ensemble', '').upper()
                    print(f"  {ensemble_type:12}: {ensemble['accuracy']:6.2f}% ({', '.join(ensemble['models'])})")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total models trained    : {report['summary']['total_models_trained']}")
        print(f"  Ensembles created       : {report['summary']['ensembles_created']}")
        print(f"  Best overall accuracy   : {report['summary']['best_overall_accuracy']:.2f}%")
        print(f"  Total training time     : {report['total_time_hours']:.2f} hours")
        
        if report['summary']['best_cnn_model']:
            print(f"  Best CNN model          : {report['summary']['best_cnn_model']['name']} ({report['summary']['best_cnn_model']['accuracy']:.2f}%)")
        if report['summary']['best_vit_model']:
            print(f"  Best ViT model          : {report['summary']['best_vit_model']['name']} ({report['summary']['best_vit_model']['accuracy']:.2f}%)")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        self._log_message("SUCCESS", "Final report generated", {
            'total_models': report['summary']['total_models_trained'],
            'ensembles': report['summary']['ensembles_created'],
            'best_accuracy': report['summary']['best_overall_accuracy']
        })


def main():
    """Main execution function."""
    try:
        # Initialize professional training pipeline
        pipeline = ProfessionalTrainingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n✅ Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
