from bpe import BPE
from token_embeddings import TokenEmbeddings
from positional_embedings import PositionalEmbeddings
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


x = torch.tensor([[1, 5, 7, 17], [17, 5, 1, 3]])
model = TokenEmbeddings(vocab_size=20, emb_size=10)    
    
print("\nВходной тензор:")
print(x)

result = model.forward(x)

print("\nРезультат:")
print(result)


pos_emb = PositionalEmbeddings(max_seq_len=100, emb_size=64)
result = pos_emb(10)  # shape: (10, 64) - первые 10 строк