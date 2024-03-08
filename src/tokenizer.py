import tiktoken
from typing import List

class SimpleTokenizer:
    def __init__(self, chars, vocab_size) -> None:
        self.chars = chars
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        return [stoi[c] for c in text] 

    def decode(self, integers: List[int]) -> str: 
        itos = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([itos[i] for i in integers]) 

class OpenAITokenizer:
    def __init__(self, type = "gpt2") -> None:
        self.enc = tiktoken.get_encoding(type)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, lst: List[int]) -> str: 
        return ''.join([self.enc.decode([el]) for el in lst])  