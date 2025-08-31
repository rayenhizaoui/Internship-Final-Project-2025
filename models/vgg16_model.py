import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg16

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class VGG16Model(nn.Module):
    """Modern VGG-16 model with transfer learning and regularization."""
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(VGG16Model, self).__init__()
        
        # Load pre-trained VGG16
        self.backbone = vgg16(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in self.backbone.features[:20].parameters():
            param.requires_grad = False
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[0].in_features
        
        # Replace the classifier with modern architecture
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(min(dropout_rate * 1.2, 0.8)),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def create_vgg16_model(input_shape=(3, 256, 256), num_classes=3, 
                       dropout_rate=0.5, learning_rate=1e-4, 
                       optimizer='adamw', weight_decay=1e-3, momentum=0.9):
    """Create the VGG-16 model with transfer learning and modern optimization."""
    model = VGG16Model(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ VGG16 model created successfully!")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Optimizer: {optimizer}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Dropout rate: {dropout_rate}")
    print(f"   - Weight decay: {weight_decay}")
    
    # Setup optimizer with weight decay
    if optimizer == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use label smoothing for better generalization
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    return model, opt, criterion

def create_vgg16_for_random_search(**params):
    """Wrapper function for random search optimization with modern hyperparameters."""
    model, optimizer, criterion = create_vgg16_model(
        dropout_rate=params.get('dropout_rate', 0.5),
        learning_rate=params.get('learning_rate', 1e-4),
        optimizer=params.get('optimizer', 'adamw'),
        weight_decay=params.get('weight_decay', 1e-3),
        momentum=params.get('momentum', 0.9)
    )
    return model, optimizer, criterion