"""
MViTv2 Model for Corn Disease Classification
Multiscale Vision Transformer v2 with improved hierarchical attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with multi-scale support."""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x


class MultiScaleAttention(nn.Module):
    """Multi-scale attention with pooling for hierarchical features."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 pool_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.pool_size = pool_size
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Pooling for multi-scale
        if pool_size > 1:
            self.pool = nn.AvgPool1d(pool_size, stride=pool_size, 
                                   padding=pool_size // 2)
        else:
            self.pool = None
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply pooling to k and v for multi-scale
        if self.pool is not None:
            # Reshape for pooling: [B, num_heads, N, C//num_heads] -> [B*num_heads, C//num_heads, N]
            B, num_heads, N, head_dim = k.shape
            k = k.permute(0, 1, 3, 2).contiguous().view(B * num_heads, head_dim, N)
            v = v.permute(0, 1, 3, 2).contiguous().view(B * num_heads, head_dim, N)
            
            # Apply 1D pooling
            k = self.pool(k)  # [B*num_heads, head_dim, N_pooled]
            v = self.pool(v)  # [B*num_heads, head_dim, N_pooled]
            
            # Reshape back: [B*num_heads, head_dim, N_pooled] -> [B, num_heads, N_pooled, head_dim]
            N_pooled = k.size(-1)
            k = k.view(B, num_heads, head_dim, N_pooled).permute(0, 1, 3, 2).contiguous()
            v = v.view(B, num_heads, head_dim, N_pooled).permute(0, 1, 3, 2).contiguous()
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MViTBlock(nn.Module):
    """Multi-scale Vision Transformer block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., pool_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, pool_size=pool_size
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MViTv2Model(nn.Module):
    """Multiscale Vision Transformer v2 for corn disease classification."""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks with different pool sizes for multi-scale
        self.blocks = nn.ModuleList([
            MViTBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                pool_size=3 if i % 4 == 0 else 1  # Multi-scale every 4th block
            )
            for i in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate position encodings for different input sizes."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        # Interpolate
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(w0, h0),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        """Prepare tokens with position encoding."""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Use class token for classification
        return self.head(x[:, 0])


def create_mvitv2_model(num_classes=3, img_size=256, model_size='small', learning_rate=3e-4, dropout_rate=0.1, **kwargs):
    """Factory function to create MViTv2 model with optimizer and criterion."""
    
    # Model configurations
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'patch_size': 16
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'patch_size': 16
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16
        }
    }
    
    config = configs.get(model_size, configs['small'])
    
    model = MViTv2Model(
        img_size=img_size,
        patch_size=config['patch_size'],
        in_chans=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=dropout_rate,
        attn_drop_rate=dropout_rate,
        **kwargs
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def create_mvitv2_for_random_search(**params):
    """Wrapper function for random search optimization."""
    model, optimizer, criterion = create_mvitv2_model(
        learning_rate=params.get('learning_rate', 3e-4),
        dropout_rate=params.get('dropout_rate', 0.1),
        model_size=params.get('model_size', 'small')
    )
    return model, optimizer, criterion


if __name__ == "__main__":
    # Test model
    model = create_mvitv2_model(num_classes=3, img_size=256, model_size='small')
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
