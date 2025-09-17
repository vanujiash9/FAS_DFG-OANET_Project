import torch.nn as nn

class OffRealAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
    
    def forward(self, query_features, key_value_features):
        attn_output, _ = self.attn(
            query=query_features, 
            key=key_value_features, 
            value=key_value_features
        )
        return attn_output