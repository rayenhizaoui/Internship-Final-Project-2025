import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import json
import random
import time
import os

def calculate_metrics_pytorch(model, data_loader, device=None):
    """Calculate comprehensive metrics for a PyTorch model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_score_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }

def pytorch_cross_validation_score(model_func, X_train, y_train, model_params, cv_folds=3):
    """Perform cross-validation for PyTorch models."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold + 1}/{cv_folds}")
        
        try:
            # Create model, optimizer, and criterion
            model, optimizer, criterion = model_func(**model_params)
            
            # Split data for this fold
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Convert to PyTorch datasets and dataloaders
            from torch.utils.data import TensorDataset, DataLoader
            
            # Convert to tensors
            X_fold_train_tensor = torch.FloatTensor(X_fold_train).permute(0, 3, 1, 2)  # NHWC -> NCHW
            y_fold_train_tensor = torch.LongTensor(y_fold_train)  # Use LongTensor for labels
            X_fold_val_tensor = torch.FloatTensor(X_fold_val).permute(0, 3, 1, 2)
            y_fold_val_tensor = torch.LongTensor(y_fold_val)  # Use LongTensor for labels
            
            # Create datasets and dataloaders
            train_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
            val_dataset = TensorDataset(X_fold_val_tensor, y_fold_val_tensor)
            
            batch_size = model_params.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop for quick evaluation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Quick training (3 epochs for CV evaluation)
            for epoch in range(3):
                model.train()
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            fold_score = correct / total if total > 0 else 0
            scores.append(fold_score)
            
            print(f"    Fold {fold + 1} score: {fold_score:.4f}")
            
        except Exception as e:
            print(f"    Error in fold {fold + 1}: {str(e)}")
            scores.append(0.0)  # Use 0 score for failed folds
            continue
    
    return np.mean(scores)

def random_search_optimization_pytorch(model_func, param_space, X_train, y_train, 
                                     n_iterations=10, cv_folds=3, save_path=None):
    """
    Perform random search hyperparameter optimization for PyTorch models.
    
    Args:
        model_func: Function that creates and returns (model, optimizer, criterion)
        param_space: Dictionary of parameter names and their possible values
        X_train: Training data
        y_train: Training labels
        n_iterations: Number of random search iterations
        cv_folds: Number of cross-validation folds
        save_path: Path to save results
    
    Returns:
        best_params: Best hyperparameters found
        best_score: Best cross-validation score
        all_results: List of all results from random search
    """
    
    print(f"Starting PyTorch Random Search with {n_iterations} iterations and {cv_folds}-fold CV")
    
    best_score = 0.0
    best_params = None
    all_results = []
    
    for i in range(n_iterations):
        print(f"\nIteration {i + 1}/{n_iterations}")
        
        # Generate random hyperparameters
        current_params = {}
        for param_name, param_values in param_space.items():
            current_params[param_name] = random.choice(param_values)
        
        print(f"Testing parameters: {current_params}")
        
        try:
            # Perform cross-validation
            start_time = time.time()
            cv_score = pytorch_cross_validation_score(
                model_func, X_train, y_train, current_params, cv_folds
            )
            end_time = time.time()
            
            result = {
                'iteration': i + 1,
                'parameters': current_params,
                'cv_score': cv_score,
                'time_taken': end_time - start_time
            }
            
            all_results.append(result)
            
            print(f"CV Score: {cv_score:.4f} (Time: {end_time - start_time:.2f}s)")
            
            # Update best parameters if current score is better
            if cv_score > best_score:
                best_score = cv_score
                best_params = current_params.copy()
                print(f"🎯 New best score: {best_score:.4f}")
                
        except Exception as e:
            print(f"Error in iteration {i + 1}: {str(e)}")
            # Still record the failed attempt
            result = {
                'iteration': i + 1,
                'parameters': current_params,
                'cv_score': 0.0,
                'error': str(e),
                'time_taken': 0.0
            }
            all_results.append(result)
            continue
    
    # Save results if path provided
    if save_path:
        results_summary = {
            'best_parameters': best_params,
            'best_cv_score': best_score,
            'search_summary': {
                'total_iterations': n_iterations,
                'cv_folds': cv_folds,
                'successful_iterations': len([r for r in all_results if r['cv_score'] > 0]),
                'best_iteration': None
            },
            'all_results': all_results
        }
        
        # Find best iteration
        if best_params:
            for result in all_results:
                if (result['parameters'] == best_params and 
                    result['cv_score'] == best_score):
                    results_summary['search_summary']['best_iteration'] = result['iteration']
                    break
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to JSON
        try:
            with open(save_path, 'w') as f:
                json.dump(results_summary, f, indent=4, default=str)
            print(f"📄 Results saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save results to {save_path}: {e}")
    
    print(f"\n🏁 Random Search Completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    return best_params, best_score, all_results

# Legacy functions for backward compatibility
def generate_random_hyperparameters(param_space):
    """Generate random hyperparameters from the given parameter space."""
    random_params = {}
    for param, values in param_space.items():
        random_params[param] = random.choice(values)
    return random_params

def random_search_optimization(model_func, param_space, X_train, y_train, 
                             n_iterations=10, cv_folds=3, save_path=None):
    """Legacy wrapper - redirects to PyTorch version."""
    return random_search_optimization_pytorch(
        model_func, param_space, X_train, y_train, 
        n_iterations, cv_folds, save_path
    )

def calculate_metrics(model, data_loader):
    """Legacy wrapper - redirects to PyTorch version."""
    return calculate_metrics_pytorch(model, data_loader)

# Legacy metric functions (preserved for backward compatibility)
def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    """Calculate precision."""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_true, y_pred):
    """Calculate recall."""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true, y_pred):
    """Calculate F1 score."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0