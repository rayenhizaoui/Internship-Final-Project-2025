"""
DenseNet121 Model for Corn Disease Classification
Optimized PyTorch implementation with Dense connections and customizable hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss to prevent overfitting"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        target_one_hot = F.one_hot(target, n_class).float()
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -target_smooth * log_prob
        return loss.sum(dim=1).mean()



class DenseNet121Model(nn.Module):
    """
    DenseNet121 model for corn disease classification with customizable parameters.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        growth_rate (int): Growth rate for dense blocks
        pretrained (bool): Whether to use pretrained weights
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.3, growth_rate=32, pretrained=True):
        super(DenseNet121Model, self).__init__()
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace the classifier with custom head with strong regularization
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(min(dropout_rate * 1.5, 0.9)),  # Increased dropout, capped at 0.9
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(min(dropout_rate * 1.2, 0.8)),  # Increased dropout, capped at 0.8
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),  # Added extra layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, num_classes)
        )
        
        # Store hyperparameters
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier layers."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps from different layers."""
        features = {}
        
        # Initial convolution and pooling
        x = self.backbone.features.conv0(x)
        x = self.backbone.features.norm0(x)
        x = self.backbone.features.relu0(x)
        features['conv0'] = x
        
        x = self.backbone.features.pool0(x)
        
        # Dense blocks
        x = self.backbone.features.denseblock1(x)
        features['denseblock1'] = x
        x = self.backbone.features.transition1(x)
        
        x = self.backbone.features.denseblock2(x)
        features['denseblock2'] = x
        x = self.backbone.features.transition2(x)
        
        x = self.backbone.features.denseblock3(x)
        features['denseblock3'] = x
        x = self.backbone.features.transition3(x)
        
        x = self.backbone.features.denseblock4(x)
        features['denseblock4'] = x
        
        x = self.backbone.features.norm5(x)
        features['final_features'] = x
        
        return features


def create_densenet121_model(input_shape=(3, 256, 256), num_classes=3, 
                            dropout_rate=0.5, growth_rate=32, learning_rate=1e-4,  # Increased dropout, decreased LR
                            optimizer='adamw', weight_decay=5e-3):  # Increased weight_decay
    """
    Create and configure DenseNet121 model with optimizer and loss function.
    
    Args:
        input_shape (tuple): Input tensor shape
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        growth_rate (int): Growth rate for dense connections
        learning_rate (float): Learning rate for optimizer
        optimizer (str): Optimizer type ('adam', 'adamw', 'sgd')
        weight_decay (float): Weight decay for regularization
    
    Returns:
        tuple: (model, optimizer, criterion)
    """
    model = DenseNet121Model(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        growth_rate=growth_rate,
        pretrained=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Configure optimizer
    if optimizer.lower() == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == 'adamw':
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    # Loss function with label smoothing to prevent overfitting
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    print(f"✅ DenseNet121 model created successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   - Growth rate: {growth_rate}")
    print(f"   - Optimizer: {optimizer}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Dropout rate: {dropout_rate}")
    
    return model, opt, criterion


def get_model_info():
    """Get information about the DenseNet121 model."""
    return {
        'name': 'DenseNet121',
        'type': 'CNN',
        'parameters': '8.0M',
        'architecture': 'Densely Connected Network',
        'key_features': [
            'Dense connections',
            'Feature reuse',
            'Gradient flow improvement',
            'Parameter efficiency'
        ],
        'strengths': [
            'Parameter efficient',
            'Strong gradient flow',
            'Feature reuse',
            'Good performance with fewer parameters'
        ],
        'optimal_hyperparameters': {
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32],
            'dropout_rate': [0.2, 0.3, 0.4],
            'growth_rate': [12, 16, 24, 32],
            'optimizer': ['adam', 'adamw']
        }
    }


class DenseBlock(nn.Module):
    """Custom Dense Block for experimentation."""
    
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
    
    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    """Custom Dense Layer."""
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


if __name__ == "__main__":
    # Test model creation
    model, optimizer, criterion = create_densenet121_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output probabilities: {torch.softmax(output, dim=1)}")
        
        # Test feature extraction
        features = model.get_feature_maps(dummy_input)
        print(f"Feature map shapes:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
