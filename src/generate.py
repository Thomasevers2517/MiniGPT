# load model from checkpoint.
from rbloom import Bloom
import torch
from model.litmodel import LitGPT
from tokenizer.bpe import BPETokenizer
from tokenizer.openai import OpenAITokenizer
from tokenizer.simple import SimpleTokenizer

torch.manual_seed(100)

filename = 'input.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

## SIMPLE TOKENIZER
ckpt_path = 'checkpoints/best_simple.ckpt'
    
ckpt_config = {
    "batch_size": 256,
    "block_size": 64,
    "dropout": 0,
    "eval_interval": 5,
    "limit_train_batches": 64,
    "limit_val_batches": 128,
    "lr": 0.00021101179446090905,
    "max_iters": 1200,
    "min_delta_lr_factor": 60,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 4,
    "precision": 16,
    "token": "simple"
}

## OPENAI TOKENIZER 
# ckpt_path = 'checkpoints/best_openai.ckpt'
    
# ckpt_config = {
#     "batch_size": 256,
#     "block_size": 16,
#     "dropout": 0,
#     "eval_interval": 5,
#     "limit_train_batches": 64,
#     "limit_val_batches": 128,
#     "lr": 0.0000453766380245818,
#     "max_iters": 400,
#     "min_delta_lr_factor": 60,
#     "n_embd": 512,
#     "n_head": 2,
#     "n_layer": 4,
#     "precision": 16,
#     "token": "openai"
# }

# determine tokenizer
token = ckpt_config["token"]
if token == "simple":
    tokenizer = SimpleTokenizer(text)
if token == "bpe":
    tokenizer = BPETokenizer(text)
if token == "openai":
    tokenizer = OpenAITokenizer(type='gpt2')

# load model from checkpoint
ckpt_gpt = LitGPT.load_from_checkpoint(
    ckpt_path, 
    vocab_size=tokenizer.vocab_size,
    **ckpt_config
)

model = ckpt_gpt.model

prompt = "ROMEO:"
context = tokenizer.encode(prompt)
context = torch.tensor(context, dtype=torch.long).unsqueeze(0)

# encode text
tokens = tokenizer.encode(text)

# extract n-grams from tokens
def extract_ngrams(tokens, n):
    ngrams = [] 
    for i in range(len(tokens)-n+1):
        ngram = tokens[i:i+n]
        # check if 25 -> : and 198 -> \n are in ngram 
        if 25 in set(ngram) and 198 in set(ngram):
            continue
        else:
            ngrams.append(ngram)
    return ngrams

n = 7
ngrams = extract_ngrams(tokens, n)

# create a bloom filter with 1% false positive rate
bf = Bloom(expected_items=len(ngrams), false_positive_rate=0.01)

# # add ngrams to bloom filter
# for ngram in ngrams:
#     bf.add(tokenizer.decode(ngram))

generated = model.generate(context, max_new_tokens=170*4, bloom_filter=bf, n=n, tokenizer=tokenizer)[0].tolist()
print('-------\nGENERATED TEXT:\n----------\n\n', tokenizer.decode(generated))