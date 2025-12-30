# model.py
import torch
from torch import nn
from typing import Optional


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        x = x + self.pos_emb(pos)  # [B, T, D]
        return self.dropout(x)


class GEGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        x = a * self.act(b)
        x = self.dropout(x)
        return self.out(x)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.ffn = GEGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, 1, -1)
        pooled, _ = self.mha(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.ln(pooled).squeeze(1)  # [B, D]


class AdvancedTransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int = 0,
        max_len: int = 256,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = LearnedPositionalEmbedding(max_len=max_len + 1, d_model=d_model, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, d_ff, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])

        self.use_attention_pooling = use_attention_pooling
        self.pool = AttentionPooling(d_model, nhead, dropout=dropout) if use_attention_pooling else None

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if self.tok_emb.padding_idx is not None:
            with torch.no_grad():
                self.tok_emb.weight[self.tok_emb.padding_idx].fill_(0)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            T = self.max_len

        key_padding_mask = (input_ids == self.pad_id)  # [B,T] True=PAD

        x = self.tok_emb(input_ids)  # [B,T,D]

        cls = self.cls_token.expand(B, 1, -1)  # [B,1,D]
        x = torch.cat([cls, x], dim=1)         # [B,T+1,D]

        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=input_ids.device)
        key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)  # [B,T+1]

        x = self.pos_emb(x)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        pooled = self.pool(x, key_padding_mask=key_padding_mask) if self.pool is not None else x[:, 0, :]
        return self.head(pooled)
