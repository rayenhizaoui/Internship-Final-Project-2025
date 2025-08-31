"""
Simple MobileViT Model for Corn Disease Classification
Simplified Vision Transformer without complex tensor operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMobileViTBlock(nn.Module):
    """Simplified MobileViT block with basic attention."""
    
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Local representation using convolutions
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Global representation using simple attention
        self.global_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)
        
    def forward(self, x):
        # Local representation
        local_rep = self.local_conv(x)
        
        # Global representation with attention
        global_rep = self.global_conv(x)
        attention_weights = self.attention(global_rep)
        global_rep = global_rep * attention_weights
        
        # Fusion
        fused = torch.cat([local_rep, global_rep], dim=1)
        output = self.fusion(fused)
        
        return output


class SimpleMobileViTModel(nn.Module):
    """Simplified MobileViT model for image classification."""
    
    def __init__(self, num_classes=3, in_channels=3, dropout_rate=0.1):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # MobileViT blocks
        self.block1 = SimpleMobileViTBlock(64, 96, num_heads=4)
        self.down1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 2, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        
        self.block2 = SimpleMobileViTBlock(96, 144, num_heads=6)
        self.down2 = nn.Sequential(
            nn.Conv2d(144, 144, 3, 2, 1),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        
        self.block3 = SimpleMobileViTBlock(144, 192, num_heads=8)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(192, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # MobileViT blocks
        x = self.block1(x)
        x = self.down1(x)
        
        x = self.block2(x)
        x = self.down2(x)
        
        x = self.block3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def create_mobilevit_simple_v2_model(learning_rate=3e-4, dropout_rate=0.1, 
                                    mlp_dim=384, num_heads=4, patch_size=2):
    """Create simplified MobileViT model with configurable hyperparameters."""
    model = SimpleMobileViTModel(num_classes=3, dropout_rate=dropout_rate)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def create_mobilevit_simple_v2_for_random_search(**params):
    """Wrapper function for random search optimization."""
    model, optimizer, criterion = create_mobilevit_simple_v2_model(
        learning_rate=params.get('learning_rate', 3e-4),
        dropout_rate=params.get('dropout_rate', 0.1),
        mlp_dim=params.get('mlp_dim', 384),
        num_heads=params.get('num_heads', 4),
        patch_size=params.get('patch_size', 2)
    )
    return model, optimizer, criterion


if __name__ == "__main__":
    # Test model
    model = create_mobilevit_simple_v2_model()[0]
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
