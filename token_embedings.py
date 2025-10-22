import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int):
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be int")

        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.text: str = ""
        self.tokens: List[str] = []
