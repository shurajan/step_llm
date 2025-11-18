from bpe import BPE
from token_embeddings import TokenEmbeddings
from positional_embedings import PositionalEmbeddings
from attention import HeadAttention
import torch

def main():
    torch.manual_seed(0)

    emb_size = 4
    head_size = 2
    max_seq_len = 8

    model = HeadAttention(emb_size, head_size, max_seq_len)

    batch_size = 1
    seq_len = 4

    x = torch.arange(batch_size * seq_len * emb_size, dtype=torch.float32)
    x = x.view(batch_size, seq_len, emb_size)

    print("x.shape:", x.shape)
    print("x:\n", x)

    out = model(x)

    print("\nout.shape:", out.shape)
    print("out:\n", out)


if __name__ == "__main__":
    main()


# # 🔹 Пример использования
# bpe = BPE(vocab_size=31)
# bpe.fit(
#     "Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала."
# )
# encoded = bpe.encode(
#     "Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала."
# )
# print(encoded)

# print(bpe.decode(encoded))

# bpe.save("data/bpe.dill")
# bpe2 = BPE.load("data/bpe.dill")

# print(bpe2.tokens)


# x = torch.tensor([[1, 5, 7, 17], [17, 5, 1, 3]])
# model = TokenEmbeddings(vocab_size=20, emb_size=10)    
    
# print("\nВходной тензор:")
# print(x)

# result = model.forward(x)

# print("\nРезультат:")
# print(result)


# pos_emb = PositionalEmbeddings(max_seq_len=100, emb_size=64)
# result = pos_emb(10)  # shape: (10, 64) - первые 10 строк



# torch.manual_seed(0)

# emb_size = 4
# head_size = 2
# max_seq_len = 4
# seq_len = 4
# batch_size = 1

# attn = HeadAttention(emb_size, head_size, max_seq_len)

# # простой вход — числа по порядку, чтобы не было рандомного шума
# x = torch.arange(batch_size * seq_len * emb_size, dtype=torch.float32).reshape(batch_size, seq_len, emb_size)

# out, att_w, scores = attn(x)

# print("Вход x:")
# print(x)
# print("\nМаска (используемая часть):")
# print(attn.mask[:seq_len, :seq_len])

# print("\nСырые attention scores после скейлинга (один батч):")
# print(scores[0])

# print("\nAttention weights (после softmax):")
# print(att_w[0])