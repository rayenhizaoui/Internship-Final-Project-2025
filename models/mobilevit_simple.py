"""
MobileViT Model for Corn Disease Classification
Simple Vision Transformer optimized for mobile/edge devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MobileViTBlock(nn.Module):
    """MobileViT block with local and global representation."""
    
    def __init__(self, in_channels, out_channels, patch_size=2, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # Local representation using convolutions
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Global representation using transformer
        self.global_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Transformer components
        self.norm1 = nn.LayerNorm(out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # MLP
        mlp_dim = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, out_channels),
            nn.Dropout(0.1)
        )
        
        # Fusion
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)
        
    def forward(self, x):
        # Local representation
        local_rep = self.local_conv(x)
        
        # Global representation
        B, C, H, W = x.shape
        global_rep = self.global_conv(x)
        
        # Unfold for transformer
        P = self.patch_size
        if H % P != 0 or W % P != 0:
            # Pad if needed
            pad_h = (P - H % P) % P
            pad_w = (P - W % P) % P
            global_rep = F.pad(global_rep, (0, pad_w, 0, pad_h))
            _, _, H_new, W_new = global_rep.shape
        else:
            H_new, W_new = H, W
            
        patches = global_rep.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(B, C, -1, P*P).permute(0, 2, 3, 1)
        patches = patches.contiguous().view(B * patches.size(1), P*P, C)
        
        # Transformer
        patches = self.norm1(patches)
        attn_out, _ = self.attention(patches, patches, patches)
        patches = patches + attn_out
        
        patches = self.norm2(patches)
        mlp_out = self.mlp(patches)
        patches = patches + mlp_out
        
        # Fold back
        num_patches = H_new // P * W_new // P
        patches = patches.contiguous().view(B, num_patches, P*P, C)
        patches = patches.permute(0, 3, 1, 2).contiguous().view(B, C, num_patches, P*P)
        patches = patches.contiguous().view(B, C, H_new//P, W_new//P, P, P)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        global_rep = patches.contiguous().view(B, C, H_new, W_new)
        
        # Crop back to original size if padded
        if H_new != H or W_new != W:
            global_rep = global_rep[:, :, :H, :W]
        
        # Fusion
        fused = torch.cat([local_rep, global_rep], dim=1)
        output = self.fusion(fused)
        
        return output


class MobileViTModel(nn.Module):
    """MobileViT model for image classification."""
    
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
        self.block1 = MobileViTBlock(64, 96, patch_size=2, num_heads=4)
        self.down1 = nn.Conv2d(96, 96, 3, 2, 1)
        
        self.block2 = MobileViTBlock(96, 144, patch_size=2, num_heads=6)
        self.down2 = nn.Conv2d(144, 144, 3, 2, 1)
        
        self.block3 = MobileViTBlock(144, 192, patch_size=2, num_heads=8)
        
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


def create_mobilevit_simple_model(learning_rate=3e-4, dropout_rate=0.1, 
                                 mlp_dim=384, num_heads=4, patch_size=2):
    """Create MobileViT model with configurable hyperparameters."""
    model = MobileViTModel(num_classes=3, dropout_rate=dropout_rate)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def create_mobilevit_simple_for_random_search(**params):
    """Wrapper function for random search optimization."""
    model, optimizer, criterion = create_mobilevit_simple_model(
        learning_rate=params.get('learning_rate', 3e-4),
        dropout_rate=params.get('dropout_rate', 0.1),
        mlp_dim=params.get('mlp_dim', 384),
        num_heads=params.get('num_heads', 4),
        patch_size=params.get('patch_size', 2)
    )
    return model, optimizer, criterion


if __name__ == "__main__":
    # Test model
    model = create_mobilevit_simple_model()[0]
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
