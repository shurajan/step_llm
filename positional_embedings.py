import torch
import torch.nn as nn


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()

        if not isinstance(max_seq_len, int):
            raise TypeError("max_seq_len must be int")
        if not isinstance(emb_size, int):
            raise TypeError("emb_size must be int")

        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.max_seq_len, self.emb_size)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.embedding.weight[:seq_len]

