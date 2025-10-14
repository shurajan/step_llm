from collections import Counter
from typing import Dict, List
import dill


class BPE:
    def __init__(self, vocab_size: int):
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be int")

        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.text: str = ""
        self.tokens: List[str] = []

    def fit(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be str")

        self.text = text

        tokens = set(text)
        self.tokens = sorted(tokens)

        all_tokens = list(text)

        while len(self.tokens) < self.vocab_size and len(all_tokens) > 1:
            pairs = list(zip(all_tokens, all_tokens[1:]))
            counter = Counter(pairs)

            most_common_pair, freq = counter.most_common(1)[0]
            new_token = most_common_pair[0] + most_common_pair[1]
            self.tokens.append(new_token)
            temp_tokens = []
            i = 0
            while i < len(all_tokens):
                if (
                    all_tokens[i] == most_common_pair[0]
                    and all_tokens[i + 1] == most_common_pair[1]
                ):
                    temp_tokens.append(new_token)
                    i += 2
                else:
                    temp_tokens.append(all_tokens[i])
                    i += 1

            all_tokens = temp_tokens

        self.token2id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.tokens)}

    def encode(self, text: str) -> List[int]:
        temp_tokens = []
        i = 0

        while i < len(text):
            matched = [token for token in self.tokens if text[i:].startswith(token)]

            if matched:
                token = max(matched, key=len)
                temp_tokens.append(token)
                i += len(token)
            else:
                temp_tokens.append(text[i])
                i += 1

        ids = [self.token2id[token] for token in temp_tokens]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id2token[id] for id in token_ids]
        return "".join(tokens)

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)
                
        print(f"Объект загружен из {filename}")
        return obj


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
