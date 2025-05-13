import torch
import torch.nn as nn
from einops import rearrange

class ViTWithPruning(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=128, 
                 depth=6, heads=4, mlp_dim=256, prune_layers=[3, 5], keep_ratios=[0.7, 0.5]):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.prune_layers = prune_layers
        self.keep_ratios = keep_ratios
        
        # Patch embedding
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(patch_size**2 * 3, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
            for _ in range(depth)
        ])
        
        # Token scorer
        self.token_scorer = TokenScorer(dim)
        
        # Classifier
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, _, h, w = x.shape
        p = self.patch_size
        
        # Patch embedding
        x = rearrange(x, 'b c (h1 p1) (w1 p2) -> b (h1 w1) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        # Keep track of token masks
        token_mask = torch.ones(b, x.shape[1], 1, device=x.device)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Prune tokens at specified layers
            if i in self.prune_layers:
                idx = self.prune_layers.index(i)
                keep_ratio = self.keep_ratios[idx]
                
                # Score tokens (excluding cls token)
                scores = self.token_scorer(x[:, 1:])
                scores = torch.cat([torch.ones(b, 1, 1, device=x.device), scores], dim=1)
                
                # Keep top-k tokens
                k = int(keep_ratio * x.shape[1])
                _, topk_indices = torch.topk(scores.squeeze(), k, dim=1)
                x = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.dim))
                token_mask = torch.gather(token_mask, 1, topk_indices)
        
        # Classifier
        x = x.mean(dim=1) if token_mask.sum() == 0 else (x * token_mask).sum(dim=1) / token_mask.sum()
        return self.mlp_head(x)
