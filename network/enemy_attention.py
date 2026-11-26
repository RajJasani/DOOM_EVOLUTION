"""Enemy Attention Module for variable numbers of visible entities.

Handles variable entity counts, entity type embeddings, kinematic features,
and padding mask handling according to docs/CONTRACT.md §2.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig


class EnemyAttentionModule(nn.Module):
    """Processes visible entities using entity embeddings and Multi-Head Attention.

    Guarantees robustness when 0 enemies/entities are visible (all padded),
    preventing NaN gradients and attention collapse.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.max_entities = config.max_entities
        self.embed_dim = config.attention_embed_dim
        self.num_heads = config.attention_num_heads

        # 12 entity types: 0 is pad, 1-3 enemies, 4-10 items/projectiles, 11 other
        self.type_embedding = nn.Embedding(config.num_entity_types, config.entity_type_embed_dim)

        # Entity token MLP: 5 continuous kinematic features + entity type embedding
        in_feat_dim = config.entity_features_dim + config.entity_type_embed_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(in_feat_dim, config.entity_token_dim),
            nn.LayerNorm(config.entity_token_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(config.entity_token_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity(),
        )

        # Self-attention among visible entities
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity()

        # Feed-forward block for entities
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )
        self.norm2 = nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity()

        # Learned query token for attentive pooling over visible entities
        self.query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Cross-attention pooling
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.norm_pool = nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        entity_type: torch.Tensor,
        mask: torch.Tensor,
        context_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            features: Tensor of shape (..., MAX_ENTITIES, 5)
            entity_type: Tensor of shape (..., MAX_ENTITIES), integer classes in [0, 11]
            mask: Tensor of shape (..., MAX_ENTITIES), binary {0, 1} (1 = valid entity)
            context_query: Optional external query tensor of shape (..., embed_dim)
        Returns:
            Tensor of shape (..., attention_embed_dim) representing the attended entity context
        """
        leading_shape = features.shape[:-2]
        K = self.max_entities

        feats_flat = features.reshape(-1, K, self.config.entity_features_dim)
        types_flat = entity_type.reshape(-1, K).long()
        mask_flat = mask.reshape(-1, K).float()
        batch_size = feats_flat.shape[0]

        # 1. Embed types and project kinematic features into entity tokens
        types_emb = self.type_embedding(types_flat)  # (B, K, type_embed_dim)
        raw_tokens = torch.cat([feats_flat, types_emb], dim=-1)  # (B, K, 5 + type_embed_dim)
        tokens = self.token_encoder(raw_tokens)  # (B, K, embed_dim)

        # 2. Robust padding mask handling
        # In PyTorch MultiheadAttention, key_padding_mask=True means IGNORE/MASK OUT.
        bool_mask = mask_flat > 0.5  # True for valid entities
        has_any_entity = bool_mask.any(dim=-1)  # (B,) boolean

        # Safe key_padding_mask: if all entities in a sample are 0, unmask the 0th position
        # temporarily so softmax never sees all -inf. We will zero out the final output.
        safe_key_padding_mask = ~bool_mask
        if not has_any_entity.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_any_entity, 0] = False

        # 3. Self-Attention among visible entities
        # Zero out padding tokens before attention
        tokens = tokens * mask_flat.unsqueeze(-1)
        attn_out, _ = self.self_attn(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=safe_key_padding_mask,
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        tokens = tokens * mask_flat.unsqueeze(-1)

        # 4. Attention Pooling with Query
        q = self.query_token.expand(batch_size, 1, -1)
        if context_query is not None:
            # Condition query with player context
            q = q + context_query.reshape(-1, 1, self.embed_dim)

        pooled_out, _ = self.pool_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=safe_key_padding_mask,
        )
        out = self.norm_pool(pooled_out.squeeze(1))  # (B, embed_dim)

        # 5. Mask out representations for samples that had zero visible entities
        out = torch.where(has_any_entity.unsqueeze(-1), out, torch.zeros_like(out))

        # Restore leading batch / sequence dimensions
        if leading_shape:
            out = out.reshape(*leading_shape, self.embed_dim)
        return out
