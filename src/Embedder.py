class Embedder():

    def __init__(self, model, text : str) -> None:
        self.model = model
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        pass
    
    def encode(self, text : str):
        
        if self.model == 'simple':
            return self.encode_simple(text)
        else:
            raise ValueError('Invalid model name.')
        
    def decode(self, encoded: str):
        if self.model == 'simple':
            return self.decode_simple(encoded)
        else:
            raise ValueError('Invalid model name.')
        
    def encode_simple(self, text : str):
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        encoded = [stoi[c] for c in text] # encoder: take a string, output a list of integers
        return encoded
    
    def decode_simple(self, encoded: str):
        itos = { i:ch for i,ch in enumerate(self.chars) }
        decoded = ''.join([itos[i] for i in encoded]) # decoder: take a list of integers, output a string
        return decoded
        