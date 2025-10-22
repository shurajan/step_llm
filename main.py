from bpe import BPE

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
