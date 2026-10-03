"""Encoder-decoder Transformer for image captioning (teacher forcing).

Encoder: learned pos-enc + N x (LayerNorm -> self-attn + residual -> LayerNorm
-> MLP + residual) over projected region features (B,N,2048). Region counts
vary per image (butd adaptive mode), so collate zero-pads features to the
batch max N and feat_mask (1 = real region) masks padded positions out of the
encoder self-attention and decoder cross-attention. Decoder: learned pos-enc +
N x (LayerNorm -> causal self-attn + residual -> LayerNorm -> cross-attn on
encoder out + residual -> LayerNorm -> MLP + residual), then LayerNorm ->
Linear -> logits (B,T,V).

Dropout goes through Tensor.dropout, which is a no-op unless the training loop
runs inside `Context(TRAINING=1)` (tinygrad's training flag). cap_mask is a
contract placeholder: caption padding needs no masking — the causal mask
covers real tokens, and pad positions are excluded from the loss via
ignore_index=0.
"""
import math

from tinygrad import Tensor, nn


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        self.d_model, self.n_heads, self.dropout = d_model, n_heads, dropout
        self.wq, self.wk, self.wv = (nn.Linear(d_model, d_model) for _ in range(3))
        self.wo = nn.Linear(d_model, d_model)

    def split(self, x: Tensor, B: int) -> Tensor:
        h = self.d_model // self.n_heads
        return x.reshape(B, -1, self.n_heads, h).transpose(1, 2)  # (B,H,T,h)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False,
                 mask: Tensor | None = None) -> Tensor:
        B = q.shape[0]
        q, k, v = self.split(self.wq(q), B), self.split(self.wk(k), B), self.split(self.wv(v), B)
        # attn_mask (B,1,1,N) broadcasts over heads and query length; True attends
        attn_mask = mask.reshape(B, 1, 1, -1) if mask is not None else None
        out = q.scaled_dot_product_attention(k, v, attn_mask=attn_mask,
                                             is_causal=is_causal, dropout_p=self.dropout)
        return self.wo(out.transpose(1, 2).reshape(B, -1, self.d_model))


class MLP:
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout = dropout

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).gelu().dropout(self.dropout))


class EncoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def __call__(self, x: Tensor, feat_mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=feat_mask)
        return x + self.mlp(self.norm2(x))


class DecoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def __call__(self, x: Tensor, enc: Tensor, feat_mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, is_causal=True)
        h = self.norm2(x)
        x = x + self.cross_attn(h, enc, enc, mask=feat_mask)
        return x + self.mlp(self.norm3(x))


class Transformer:
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1,
                 max_len: int = 21):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.feat_proj = nn.Linear(2048, d_model)
        self.max_enc_pos = 512  # > any num_regions
        self.enc_pos = nn.Embedding(self.max_enc_pos, d_model)
        self.enc_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.dec_embed = nn.Embedding(vocab_size, d_model)
        self.dec_pos = nn.Embedding(max_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def positions(self, length: int) -> Tensor:
        return Tensor.arange(length).reshape(1, -1)

    def __call__(self, batch) -> Tensor:
        return self.forward(batch)

    def forward(self, batch) -> Tensor:
        # model reads only the fields it needs; dataset-specific fields may
        # vary per model without touching the trainer
        feat, cap_ids = batch.feats, batch.cap_ids
        feat_mask = batch.feat_mask
        x = self.feat_proj(feat) + self.enc_pos(self.positions(feat.shape[1]))
        for layer in self.enc_layers:
            x = layer(x, feat_mask)
        y = self.dec_embed(cap_ids) * math.sqrt(self.d_model) \
            + self.dec_pos(self.positions(cap_ids.shape[1]))
        for layer in self.dec_layers:
            y = layer(y, x, feat_mask)
        return self.out(self.norm(y))