from bpe import BPE
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)

# простая проверка вычислений на MPS
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = (x @ y).sum()
z.backward() if z.requires_grad else None
print("OK")

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

bpe.save('data/bpe.dill')
bpe2 = BPE.load('data/bpe.dill')

print(bpe2.tokens)

# print(bpe.get_token_to_id())
# print(bpe.get_id_to_token())
