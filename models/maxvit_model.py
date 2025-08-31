import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MultiheadAttention

IMG_SIZE = 256  # Image size for MobileViT
LEARNING_RATE = 3e-4  # Learning rate for the optimizer

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InvertedResidualBlock(nn.Module):
    """Inverted residual block"""
    def __init__(self, in_channels, expanded_channels, out_channels, stride=1):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        if expanded_channels != in_channels:
            self.expand = ConvBlock(in_channels, expanded_channels, 1)
        else:
            self.expand = None
            
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, 3, 
                                  stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(expanded_channels)
        
        self.pointwise = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        
        if self.expand is not None:
            x = self.expand(x)
            
        x = F.relu(self.bn_dw(self.depthwise(x)))
        x = self.bn_pw(self.pointwise(x))
        
        if self.use_residual:
            x = x + residual
            
        return x

class MobileViTBlock(nn.Module):
    """MobileViT block with local representation and transformer"""
    def __init__(self, in_channels, num_heads, mlp_dim, dropout_rate=0.1):
        super(MobileViTBlock, self).__init__()
        self.local_rep = ConvBlock(in_channels, in_channels, 3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, in_channels)
        )
        
    def forward(self, x):
        local_rep = self.local_rep(x)
        B, C, H, W = local_rep.shape
        
        # Global average pooling
        global_rep = self.global_pool(local_rep).view(B, C)
        
        # MLP transformation
        global_rep = self.mlp(global_rep)
        
        # Reshape and add to local representation
        global_rep = global_rep.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        return local_rep + global_rep

class MobileViTModel(nn.Module):
    """MobileViT model for corn disease classification"""
    def __init__(self, num_classes=3, mlp_dim=384, num_heads=4, dropout_rate=0.1):
        super(MobileViTModel, self).__init__()
        
        # Initial convolution
        self.conv_stem = ConvBlock(3, 16, 3, stride=2)
        
        # MobileNet blocks
        self.block1 = InvertedResidualBlock(16, 64, 32)
        self.block2 = InvertedResidualBlock(32, 128, 32)
        self.block3 = InvertedResidualBlock(32, 128, 64, stride=2)
        self.block4 = InvertedResidualBlock(64, 256, 64)
        self.block5 = InvertedResidualBlock(64, 256, 64)
        self.block6 = InvertedResidualBlock(64, 256, 96, stride=2)
        
        # MobileViT blocks
        self.mobilevit1 = MobileViTBlock(96, num_heads, mlp_dim, dropout_rate)
        self.block7 = InvertedResidualBlock(96, 384, 128, stride=2)
        self.mobilevit2 = MobileViTBlock(128, num_heads, 512, dropout_rate)
        self.block8 = InvertedResidualBlock(128, 512, 160, stride=2)
        self.mobilevit3 = MobileViTBlock(160, num_heads, mlp_dim, dropout_rate)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(160, num_classes)
        
    def forward(self, x):
        x = self.conv_stem(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.mobilevit1(x)
        x = self.block7(x)
        x = self.mobilevit2(x)
        x = self.block8(x)
        x = self.mobilevit3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_mobilevit_model(learning_rate=3e-4, dropout_rate=0.1, 
                          mlp_dim=384, num_heads=4):
    """Create MobileViT model with configurable hyperparameters."""
    model = MobileViTModel(num_classes=3, mlp_dim=mlp_dim, 
                          num_heads=num_heads, dropout_rate=dropout_rate)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def create_mobilevit_for_random_search(**params):
    """Wrapper function for random search optimization."""
    model, optimizer, criterion = create_mobilevit_model(
        learning_rate=params.get('learning_rate', 3e-4),
        dropout_rate=params.get('dropout_rate', 0.1),
        mlp_dim=params.get('mlp_dim', 384),
        num_heads=params.get('num_heads', 4)
    )
    return model, optimizer, criterion