"""
DeiT3 Model for Corn Disease Classification
Data-efficient Image Transformer v3 with improved training strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
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


class Attention(nn.Module):
    """Multi-head self attention with improved implementation."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    """Layer scaling for improved training stability."""
    
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MLP(nn.Module):
    """MLP block with improved design."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """DeiT3 transformer block with LayerScale and improved design."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., init_values=1e-5, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DeiT3Model(nn.Module):
    """Data-efficient Image Transformer v3 for corn disease classification."""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 init_values=1e-5, use_distillation=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_distillation = use_distillation
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 2
        else:
            self.num_tokens = 1
            
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                init_values=init_values
            )
            for _ in range(depth)
        ])
        
        # Classification head(s)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        if use_distillation:
            self.head_dist = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        if use_distillation:
            nn.init.trunc_normal_(self.dist_token, std=.02)
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
        npatch = x.shape[1] - self.num_tokens
        N = self.pos_embed.shape[1] - self.num_tokens
        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, :self.num_tokens]
        patch_pos_embed = self.pos_embed[:, self.num_tokens:]
        dim = x.shape[-1]
        
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        # Add a small number to avoid floating point error
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(int(w0), int(h0)),
            mode='bicubic',
            align_corners=False,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        """Prepare tokens with position encoding."""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        # Add special tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if self.use_distillation:
            # Return both cls and dist token outputs for distillation
            x_cls = self.head(x[:, 0])
            x_dist = self.head_dist(x[:, 1])
            return x_cls, x_dist
        else:
            # Use class token for classification
            return self.head(x[:, 0])


def create_deit3_model(num_classes=3, img_size=256, model_size='small', 
                       learning_rate=3e-4, dropout_rate=0.1, use_distillation=False, **kwargs):
    """Factory function to create DeiT3 model with optimizer and criterion."""
    
    # Model configurations
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'patch_size': 16,
            'init_values': 1e-6
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'patch_size': 16,
            'init_values': 1e-6
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16,
            'init_values': 1e-6
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'patch_size': 16,
            'init_values': 1e-6
        }
    }
    
    config = configs.get(model_size, configs['small'])
    
    # Override num_heads if provided in kwargs
    num_heads = kwargs.get('num_heads', config['num_heads'])
    mlp_ratio = kwargs.get('mlp_ratio', 4.)
    patch_size = kwargs.get('patch_size', config['patch_size'])
    
    model = DeiT3Model(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=dropout_rate,
        attn_drop_rate=dropout_rate,
        init_values=config['init_values'],
        use_distillation=use_distillation
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def create_deit3_for_random_search(**params):
    """Wrapper function for random search optimization."""
    model, optimizer, criterion = create_deit3_model(
        learning_rate=params.get('learning_rate', 3e-4),
        dropout_rate=params.get('dropout_rate', 0.1),
        model_size=params.get('model_size', 'small'),
        num_heads=params.get('num_heads', 6),
        mlp_ratio=params.get('mlp_ratio', 4),
        patch_size=params.get('patch_size', 16)
    )
    return model, optimizer, criterion


if __name__ == "__main__":
    # Test model
    model = create_deit3_model(num_classes=3, img_size=256, model_size='small')
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
        
    # Test distillation model
    print("\n--- Testing with distillation ---")
    model_dist = create_deit3_model(num_classes=3, img_size=256, model_size='small', 
                                   use_distillation=True)
    
    with torch.no_grad():
        cls_out, dist_out = model_dist(x)
        print(f"Cls output shape: {cls_out.shape}")
        print(f"Dist output shape: {dist_out.shape}")
