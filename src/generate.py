# load model from checkpoint.
import torch
from model.litmodel import LitGPT
from tokenizer.bpe import BPETokenizer
from tokenizer.openai import OpenAITokenizer
from tokenizer.simple import SimpleTokenizer

filename = 'input.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

ckpt_path = 'checkpoints/epoch=164-step=10560.ckpt'
    
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

token = ckpt_config["token"]

# determine tokenizer
if token == "simple":
    tokenizer = SimpleTokenizer(text)
if token == "bpe":
    tokenizer = BPETokenizer(text)
if token == "openai":
    tokenizer = OpenAITokenizer(type='gpt2')

ckpt_gpt = LitGPT.load_from_checkpoint(
    ckpt_path, 
    vocab_size=tokenizer.vocab_size,
    **ckpt_config
)

model = ckpt_gpt.model

context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=2000)[0].tolist()
print(tokenizer.decode(generated))