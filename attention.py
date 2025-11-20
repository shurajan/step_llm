import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size: int = emb_size
        self.head_size: int = head_size
        self.max_seq_len = max_seq_len

        self.w_k = nn.Linear(self.emb_size, self.head_size)
        self.w_q = nn.Linear(self.emb_size, self.head_size)
        self.w_v = nn.Linear(self.emb_size, self.head_size)

        self.tril = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len)))

    def forward(self, x: torch.Tensor):
        K: torch.Tensor = self.w_k(x)
        Q: torch.Tensor = self.w_q(x)
        V: torch.Tensor = self.w_v(x)

        attention: torch.Tensor = Q @ K.transpose(-2, -1) / math.sqrt(self.head_size)

        attention_masked = attention.masked_fill(
            self.tril[0 : attention.shape[-1], 0 : attention.shape[-1]] == 0,
            float("-inf"),
        )

        attention_soft = F.softmax(attention_masked, dim=-1)

        return attention_soft @ V


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads: int = num_heads

        self.heads = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )

        self.out = nn.Linear(head_size * num_heads, emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([head.forward(x) for head in self.heads], dim=2)

        return self.dropout(self.out(out))
