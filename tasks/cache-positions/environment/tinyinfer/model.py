"""A small decoder-only transformer with a key/value cache.

Batches are left-padded: real tokens sit at the right edge of each row and
`attention_mask` is False on the padding. Position ids count real tokens only,
so a row produces the same logits whether it is run alone or inside a padded
batch.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    vocab_size: int = 259  # 256 byte values + BOS, EOS, PAD
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    max_seq_len: int = 48


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, key_mask, past=None):
        """x: (B, T, D) new positions. key_mask: (B, P + T) over cached + new keys."""
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        S = k.shape[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(D // self.n_head)
        # Query i sits at absolute index (S - T) + i and may see keys up to itself.
        causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        allowed = causal[None, None] & key_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        att = scores.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, key_mask, past=None):
        a, present = self.attn(self.ln1(x), key_mask, past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class TinyLM(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_emb = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size)

    def forward(self, tokens, attention_mask=None, past=None):
        """Run the model on `tokens` (B, T).

        attention_mask: bool (B, P + T) covering cached positions followed by
            the new ones; False marks left padding. Defaults to all True.
        past: per-layer (k, v) cache returned by an earlier call, or None.

        Returns (logits (B, T, vocab), present) where present is the updated cache.
        """
        B, T = tokens.shape
        P = past[0][0].shape[2] if past is not None else 0
        if P + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {P + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        if attention_mask is None:
            attention_mask = torch.ones(B, P + T, dtype=torch.bool, device=tokens.device)
        # Position of each token = number of real tokens before it in its row.
        if past is None:
            positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        else:
            # Cached steps continue from the end of the cache.
            positions = torch.arange(P, P + T, device=tokens.device).expand(B, T)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        present = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, attention_mask, past[i] if past is not None else None)
            present.append(kv)
        return self.head(self.ln_f(x)), present
