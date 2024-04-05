import sys
sys.path.insert(0, '/Users/rodrigoalvarezlucendo/Desktop/MiniGPT')

import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
from src.model.decoder import Block


class BigramLanguageModel(nn.Module):

    def __init__(
            self, 
            vocab_size,
            block_size=16, 
            n_embd=32,
            n_head=4,
            n_layer=4,
            dropout=0.0
        ):
        super(BigramLanguageModel, self).__init__()
        
        self.block_size = block_size
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T).to(tok_emb).long()) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)        
        return logits

    def generate(self, idx, max_new_tokens, bloom_filter, n, tokenizer):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
    
            for _ in range(3):
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                
                # check if the n-gram is in the bloom filter
                idx_cand = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                ngram = idx_cand[:, -n:]
                ngram = ngram[0].detach().numpy()
                ngram = tokenizer.decode(ngram)
                if ngram in bloom_filter:
                    # print(f"Found ngram: {ngram}")
                    # change the probability of this token
                    probs[0, idx_next[0]] = 0
                else:
                    # add the new token to the prefix
                    idx = idx_cand
                    break    
        return idx
    
    