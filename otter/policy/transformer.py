import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from otter.util.args import ModelConfig
from timm.layers.mlp import Mlp
import numpy as np
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Implements Sinusoidal Positional Encoding (used in Transformer).
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))

        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE).
    Reference: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        seq_len, batch_size, dim = x.shape
        assert dim == self.embed_dim, "Embedding dimension mismatch!"

        # Compute RoPE embeddings
        theta = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
        m = torch.arange(seq_len).float().unsqueeze(1) * theta.unsqueeze(0)
        sin, cos = torch.sin(m), torch.cos(m)

        # Reshape for broadcasting
        sin = sin.unsqueeze(1)  # (seq_len, 1, dim//2)
        cos = cos.unsqueeze(1)  # (seq_len, 1, dim//2)

        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even/odd
        x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rope

class MultiHeadAttention(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError(
                f"Hidden size {config.transformer_dim} is not divisible by number of attention heads {config.transformer_heads}"
            )
        self.transformer_heads = config.transformer_heads
        self.attention_head_size = int(config.transformer_dim / config.transformer_heads)
        self.all_head_size = self.transformer_heads * self.attention_head_size

        self.query = nn.Linear(config.transformer_dim, self.all_head_size)
        self.key = nn.Linear(config.transformer_dim, self.all_head_size)
        self.value = nn.Linear(config.transformer_dim, self.all_head_size)

        self.dropout = config.attention_probs_dropout_prob
        self.out = nn.Linear(config.transformer_dim, config.transformer_dim)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.transformer_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if attention_mask is not None:
            # attention_mask should be broadcastable to [batch, num_heads, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

        # Use torch's scaled_dot_product_attention
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.out(context_layer)
    

class TransformerBlock(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.pre_ln = nn.LayerNorm(config.transformer_dim)
        self.post_ln = nn.LayerNorm(config.transformer_dim)
        intermediate_size = config.transformer_dim * config.transformer_expansion_factor
        mlp = [
            nn.Linear(config.transformer_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, config.transformer_dim),
        ]
        self.mlp = nn.Sequential(*mlp)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        
        if attention_mask is not None:
            assert not is_causal, "Causal mask not supported with attention mask"

        attention_output = self.attention(
            self.pre_ln(hidden_states),
            attention_mask,
            is_causal,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        hidden_states = hidden_states + self.dropout(self.mlp(self.post_ln(hidden_states)))
        return hidden_states

class CausalTransformer(nn.Module):
    """Main policy transformer with RoPE and causal attention"""
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.transformer_layers)
        ])

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Forward through each transformer block
        for layer in self.layers:
            x = layer.forward(x, is_causal=True)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, kv_input_dim : int, q_dim : int, mlp_dim : int, num_heads : int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            q_dim, 
            num_heads, 
            batch_first=True, 
            kdim=kv_input_dim, 
            vdim=kv_input_dim,
        )
        self.mlp = Mlp(
            in_features=q_dim, 
            hidden_features=mlp_dim, 
            out_features=q_dim,
        )
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.kv_layer_norm = nn.LayerNorm(kv_input_dim)
        self.layer_norm = nn.LayerNorm(q_dim)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        q = self.q_layer_norm(q)
        kv = self.kv_layer_norm(kv)
        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=mask)
        q = q + attn_out  # Add residual connection
        x = self.layer_norm(q)
        x = self.mlp(x)
        return q + x

class AttentionPooling(nn.Module):
    """Attention pooling layer to compress sequence of tokens into a single token"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, num_layers: int = 2, num_readouts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_readouts = num_readouts
        self.intermediate_dim = self.output_dim // self.num_readouts
        assert self.intermediate_dim * self.num_readouts == self.output_dim, "Output dim must be divisible by num readouts"
        self.query = nn.Parameter(torch.randn(1, num_readouts, self.intermediate_dim))
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                kv_input_dim=input_dim, 
                q_dim=self.intermediate_dim, 
                mlp_dim=self.output_dim, 
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
            
        for layer in self.blocks:
            query = layer(query, x, mask)
        
        query = self.layer_norm(query)
        return query.reshape(batch_size, -1)  # dimension: [batch, output_dim]