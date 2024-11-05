import os
import wandb
import torch
from torch.utils.data import DataLoader
from model.litmodel import LitGPT
from tokenizer.simple import SimpleTokenizer
from tokenizer.openai import OpenAITokenizer
from tokenizer.bpe import BPETokenizer
from lightning.pytorch.loggers import WandbLogger
from dataset import ShakespareDataset
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler



# Set precision and seed
L.seed_everything(42)
torch.set_float32_matmul_precision('medium')
wandb.init()
wandb_logger = WandbLogger(
    log_model=True,
)

# Initialize wandb

config = wandb.config

                

# Read the input text file
filename = 'input.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()


# Determine tokenizer based on config
if config["token"] == "simple":
    tokenizer = SimpleTokenizer(text)
elif config["token"] == "bpe":
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(text)
elif config["token"] == "openai":
    tokenizer = OpenAITokenizer(type='gpt2')
else:
    raise ValueError(f"Unknown tokenizer type: {config['token']}")

# Encode text
tokens = tokenizer.encode(text)

# Split dataset into train, validation, and test
dataset = ShakespareDataset(tokens, config["block_size"])
n = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - n - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [n, val_size, test_size]
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=64,
    pin_memory=True,
    persistent_workers=True,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=64,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=8
)

# Create the model
gpt = LitGPT(vocab_size=tokenizer.vocab_size, **config)


profiler = PyTorchProfiler(
activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
],
record_shapes=True,
with_stack=True,
with_flops=True,  # Enable FLOPs counting
profile_memory=True,
schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
on_trace_ready=torch.profiler.tensorboard_trace_handler('./lightning_logs')
)

# Create the trainer
trainer = L.Trainer(
    accelerator="gpu",
    limit_train_batches=config["limit_train_batches"],
    limit_val_batches=config["limit_val_batches"],
    max_epochs=config["max_iters"],
    logger=wandb_logger,
    precision=config["precision"],  # Updated to recommended precision
    check_val_every_n_epoch=config["eval_interval"],
    callbacks=[
        EarlyStopping(
            monitor="validation_loss",
            min_delta=config["lr"] * config["min_delta_lr_factor"],
            patience=2
        )
    ],
    profiler=profiler
)

# Train the model
trainer.fit(model=gpt, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test the model
trainer.test(model=gpt, dataloaders=test_loader)

wandb.finish()

