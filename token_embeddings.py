import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()  # ← ВАЖНО! Нужно вызвать инициализацию родителя
        
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be int")
        if not isinstance(emb_size, int):
            raise TypeError("emb_size must be int")
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for row in x:
            print(row)