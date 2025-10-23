from bpe import BPE
from token_embeddings import TokenEmbeddings
import torch


# 🔹 Пример использования
bpe = BPE(vocab_size=31)
bpe.fit(
    "Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала."
)
encoded = bpe.encode(
    "Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала."
)
print(encoded)

print(bpe.decode(encoded))

bpe.save("data/bpe.dill")
bpe2 = BPE.load("data/bpe.dill")

print(bpe2.tokens)


x = torch.tensor([[113, 456, 76, 345], [345, 678, 454, 546]])
model = TokenEmbeddings(vocab_size=1000, emb_size=64)    
    
print("\nВходной тензор:")
print(x)

model.forward(x)
