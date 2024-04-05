from typing import List # type: ignore

class SimpleTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(list(set(text))) 
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        return [stoi[c] for c in text] 

    def decode(self, integers: List[int]) -> str: 
        itos = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([itos[i] for i in integers]) 