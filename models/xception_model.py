"""
Xception Model for Corn Disease Classification
Optimized PyTorch implementation with Depthwise Separable Convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception block with separable convolutions and residual connections."""
    
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(XceptionBlock, self).__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
            
        self.relu = nn.ReLU(inplace=True)
        rep = []
        
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
            
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
            
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
            
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
            
        self.rep = nn.Sequential(*rep)
        
    def forward(self, inp):
        x = self.rep(inp)
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
            
        x += skip
        return x


class XceptionModel(nn.Module):
    """
    Xception model for corn disease classification with customizable parameters.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(XceptionModel, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        
        # Middle flow
        self.block4 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        self.block8 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def create_xception_model(input_shape=(3, 256, 256), num_classes=3, 
                         dropout_rate=0.3, learning_rate=3e-4, 
                         optimizer='adam', weight_decay=1e-4, label_smoothing=0.0):
    """
    Create and configure Xception model with optimizer and loss function.
    
    Args:
        input_shape (tuple): Input tensor shape
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        optimizer (str): Optimizer type ('adam', 'adamw', 'sgd')
        weight_decay (float): Weight decay for regularization
        label_smoothing (float): Label smoothing factor
    
    Returns:
        tuple: (model, optimizer, criterion)
    """
    model = XceptionModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate
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
    
    # Loss function with optional label smoothing
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Xception model created successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   - Optimizer: {optimizer}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Dropout rate: {dropout_rate}")
    print(f"   - Label smoothing: {label_smoothing}")
    
    return model, opt, criterion


def get_model_info():
    """Get information about the Xception model."""
    return {
        'name': 'Xception',
        'type': 'CNN',
        'parameters': '22.9M',
        'architecture': 'Extreme Inception with Depthwise Separable Convolutions',
        'key_features': [
            'Depthwise separable convolutions',
            'Extreme Inception architecture',
            'Residual connections',
            'Efficient parameter usage'
        ],
        'strengths': [
            'Parameter efficient',
            'Strong feature extraction',
            'Good accuracy/complexity trade-off',
            'Robust to overfitting'
        ],
        'optimal_hyperparameters': {
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32],
            'dropout_rate': [0.3, 0.4, 0.5],
            'optimizer': ['adam', 'adamw'],
            'weight_decay': [1e-5, 1e-4],
            'label_smoothing': [0.0, 0.1, 0.2]
        }
    }


if __name__ == "__main__":
    # Test model creation
    model, optimizer, criterion = create_xception_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output probabilities: {torch.softmax(output, dim=1)}")
        
    # Model summary
    print(f"\nModel Information:")
    info = get_model_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
