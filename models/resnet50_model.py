"""
ResNet50 Model for Corn Disease Classification
Optimized PyTorch implementation with customizable hyperparameters
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



class ResNet50Model(nn.Module):
    """
    ResNet50 model for corn disease classification with customizable parameters.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone parameters
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.3, pretrained=True, freeze_backbone=False):
        super(ResNet50Model, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head with strong regularization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(min(dropout_rate * 1.5, 0.9)),  # Increased dropout, capped at 0.9
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.Dropout(min(dropout_rate * 1.2, 0.8)),  # Increased dropout, capped at 0.8
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),  # Added extra layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract features using backbone (without final fc layer)
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply custom classifier
        output = self.classifier(features)
        
        return output


def create_resnet50_model(input_shape=(3, 256, 256), num_classes=3, 
                         dropout_rate=0.5, learning_rate=1e-4,  # Increased dropout, decreased LR
                         optimizer='adamw', weight_decay=5e-3, momentum=0.9):  # Increased weight_decay
    """
    Create and configure ResNet50 model with optimizer and loss function.
    
    Args:
        input_shape (tuple): Input tensor shape
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        optimizer (str): Optimizer type ('adam', 'adamw', 'sgd')
        weight_decay (float): Weight decay for regularization
        momentum (float): Momentum for SGD optimizer
    
    Returns:
        tuple: (model, optimizer, criterion)
    """
    model = ResNet50Model(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=True,
        freeze_backbone=False
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
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    # Loss function with label smoothing to prevent overfitting
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    print(f"✅ ResNet50 model created successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   - Optimizer: {optimizer}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Dropout rate: {dropout_rate}")
    
    return model, opt, criterion


def get_model_info():
    """Get information about the ResNet50 model."""
    return {
        'name': 'ResNet50',
        'type': 'CNN',
        'parameters': '25.6M',
        'architecture': 'Deep Residual Network',
        'key_features': [
            'Residual connections',
            'Batch normalization',
            'Deep architecture (50 layers)',
            'Transfer learning ready'
        ],
        'strengths': [
            'Excellent feature extraction',
            'Robust to vanishing gradients',
            'Well-tested architecture',
            'Good generalization'
        ],
        'optimal_hyperparameters': {
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.2, 0.3, 0.4],
            'optimizer': ['adam', 'adamw'],
            'weight_decay': [1e-4, 1e-3]
        }
    }


if __name__ == "__main__":
    # Test model creation
    model, optimizer, criterion = create_resnet50_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output probabilities: {torch.softmax(output, dim=1)}")
