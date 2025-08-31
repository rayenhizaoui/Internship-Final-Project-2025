import torch
import torch.nn.functional as F
import numpy as np

def soft_voting_predict(models, X):
    """Perform soft voting on the predictions of the given models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X_tensor = X.to(device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            pred = model(X_tensor)
            # Apply softmax to get probabilities
            pred_probs = F.softmax(pred, dim=1)
            predictions.append(pred_probs.cpu().numpy())
    
    # Average the predictions
    avg_predictions = np.mean(predictions, axis=0)
    
    # Return the class with the highest average probability
    return np.argmax(avg_predictions, axis=1)

def load_models(vgg_model_path, mobilevit_model_path):
    """Load the VGG-16 and MobileViT models from the specified paths."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VGG-16 model
    from models.vgg16_model import VGG16Model
    vgg_model = VGG16Model()
    vgg_model.load_state_dict(torch.load(vgg_model_path, map_location=device))
    vgg_model = vgg_model.to(device)
    
    # Load MobileViT model
    from models.mobilevit_model import MobileViTModel
    mobilevit_model = MobileViTModel()
    mobilevit_model.load_state_dict(torch.load(mobilevit_model_path, map_location=device))
    mobilevit_model = mobilevit_model.to(device)
    
    return vgg_model, mobilevit_model

def ensemble_predict(vgg_model_path, mobilevit_model_path, X):
    """Load models and perform ensemble prediction on the input data."""
    vgg_model, mobilevit_model = load_models(vgg_model_path, mobilevit_model_path)
    models = [vgg_model, mobilevit_model]
    return soft_voting_predict(models, X)

class EnsembleVoting:
    """Ensemble voting class for combining multiple PyTorch models"""
    
    def __init__(self, models):
        self.models = models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
    
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                else:
                    X_tensor = X.to(self.device)
                
                pred = model(X_tensor)
                pred_probs = F.softmax(pred, dim=1)
                predictions.append(pred_probs.cpu().numpy())
        
        # Average predictions
        avg_predictions = np.mean(predictions, axis=0)
        return np.argmax(avg_predictions, axis=1)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                else:
                    X_tensor = X.to(self.device)
                
                pred = model(X_tensor)
                pred_probs = F.softmax(pred, dim=1)
                predictions.append(pred_probs.cpu().numpy())
        
        # Return average probabilities
        return np.mean(predictions, axis=0)