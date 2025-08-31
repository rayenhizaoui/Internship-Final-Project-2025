"""
MaxViT Model for Corn Disease Classification
Simplified Multi-axis Vision Transformer with local and global attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv for MaxViT."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expand
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class GridAttention(nn.Module):
    """Grid-based multi-head self attention."""
    
    def __init__(self, dim, grid_size=2, num_heads=8):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Grid attention - simplified version
        x_flat = x.view(B, H * W, C)
        B_, N, C = x_flat.shape
        
        qkv = self.qkv(x_flat).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_out = self.proj(x_out)
        
        return x_out.view(B, H, W, C)


class TransformerBlock(nn.Module):
    """Transformer block with window and grid attention."""
    
    def __init__(self, dim, window_size=7, grid_size=2, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.grid_size = grid_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.grid_attn = GridAttention(dim, grid_size, num_heads)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Window attention (simplified)
        shortcut = x
        x = self.norm1(x)
        x_flat = x.view(B, H * W, C)
        x_attn = self.window_attn(x_flat)
        x = shortcut + x_attn.view(B, H, W, C)
        
        # Grid attention
        x = x + self.grid_attn(self.norm2(x))
        
        # MLP
        x_flat = x.view(B, H * W, C)
        x_mlp = self.mlp(self.norm3(x_flat))
        x = x + x_mlp.view(B, H, W, C)
        
        return x


class MaxViTStage(nn.Module):
    """MaxViT stage with MBConv and Transformer blocks."""
    
    def __init__(self, in_channels, out_channels, depth=2, num_heads=8):
        super().__init__()
        
        # MBConv blocks
        self.mbconv = MBConv(in_channels, out_channels)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(out_channels, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        # Downsample
        self.downsample = nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=2, stride=2) if in_channels != out_channels else None
    
    def forward(self, x):
        # MBConv
        x = self.mbconv(x)
        
        # Transformer blocks (convert to H,W,C format)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Downsample
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x


class MaxViTModel(nn.Module):
    """Simplified MaxViT model for corn disease classification."""
    
    def __init__(self, num_classes=3, img_size=256, channels=[64, 128, 256, 512], 
                 depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32]):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = channels[0]
        
        for i, (out_channels, depth, num_head) in enumerate(zip(channels, depths, num_heads)):
            stage = MaxViTStage(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                num_heads=num_head
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def create_maxvit_model(num_classes=3, img_size=256, **kwargs):
    """Factory function to create MaxViT model."""
    model = MaxViTModel(
        num_classes=num_classes,
        img_size=img_size,
        channels=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_maxvit_model(num_classes=3, img_size=256)
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
