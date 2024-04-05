import tiktoken
from typing import List # type: ignore


class OpenAITokenizer:
    def __init__(self, type = "gpt2") -> None:      
        self.enc = tiktoken.get_encoding(type)
        self.vocab_size = self.enc.n_vocab

    def extend(self, tokens_to_ids: dict) -> None:
        special_tokens = {
            **self.enc._special_tokens,
            **tokens_to_ids
        }    
        extended_enc = tiktoken.Encoding(
            name="shakespeare",
            pat_str=self.enc._pat_str,
            mergeable_ranks=self.enc._mergeable_ranks,
            special_tokens=special_tokens
        )
        # update the tokenizer.
        self.enc = extended_enc
        # update the vocab size.
        self.vocab_size = extended_enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, tokens: List[int], special_tokens: bool = True) -> str: 
        if not special_tokens:
            special_token_ids = self.enc._special_tokens.values()
            tokens = [t for t in tokens if t not in special_token_ids]
        return "".join(self.enc.decode_batch([tokens])) 