import torch
from torch.utils.data import DataLoader
from model.litmodel import LitGPT
from tokenizer.simple import SimpleTokenizer
from tokenizer.openai import OpenAITokenizer
from tokenizer.bpe import BPETokenizer
from lightning.pytorch.loggers import WandbLogger
from dataset import ShakespareDataset
import pytorch_lightning as L
import wandb
from pytorch_lightning.callbacks import EarlyStopping

# sweep configuration.
sweep_configuration = {
    "method": "random", # grid, random, bayes
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [256]},
        "limit_train_batches": {"values": [64]},
        "max_iters": {"values": [400]},
        "lr": {"min": 1e-5, "max": 5e-4},
        "block_size": {"values": [4, 16, 64]},
        "n_embd": {"values": [8, 32, 128, 512]},
        "n_head": {"values": [2, 4, 8]},
        "n_layer": {"values": [2, 4, 8]},
        "dropout": {"values": [0.0]},
        "token": {"values": ["simple"]},
        "precision": {"values": [16]}, #can add 16 if BPE is fixed
        "eval_interval": {"values": [5]},
        "limit_val_batches": {"values": [128]},
        "min_delta_lr_factor": {"values": [60]},#{"min": 1e-2, "max": 2e-2},
    }
}

# this configuration corresponds to the best model found in the
# sweep for the simple tokenizer.
config = {
    "batch_size": 256,
    "limit_train_batches": 64,
    "max_iters": 400,
    "lr": 0.2110,
    "block_size": 64,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 4,
    "dropout": 0.0,
    "token": "simple",
    "precision": 16, 
    "eval_interval": 5,
    "limit_val_batches": 128,
    "min_delta_lr_factor": 60
}

def main():
    # set seed for reproducibility.
    torch.manual_seed(1337)

    # log to wandb.
    wandb.init(project="gpt", config=config)
    wandb_logger = WandbLogger(log_model=True)
    # config = wandb.config     # # uncomment this line if using sweep.
    
    # read file.
    filename = 'input.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # determine tokenizer.
    if config["token"] == "simple":
        tokenizer = SimpleTokenizer(text)
    if config["token"] == "bpe":
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(text)
    if config["token"] == "openai":
        tokenizer = OpenAITokenizer(type='gpt2')

    # encode text.
    tokens = tokenizer.encode(text)

    # train val test split.
    dataset = ShakespareDataset(tokens, config["block_size"])
    n = int(0.9*len(dataset)) # first 90% will be train, rest val and test 50/50
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n, int(0.05*len(dataset)), len(dataset) - n - int(0.05*len(dataset))]
    )
    
    # create dataloaders.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        num_workers=7, 
        pin_memory=True, 
        persistent_workers=True, 
        shuffle=True 
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=7, 
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=7
    )

    # create model.
    gpt = LitGPT(vocab_size=tokenizer.vocab_size, **config)
    
    # create trainer.
    trainer = L.Trainer(
        accelerator="cpu", # change to gpy if you have one.
        limit_train_batches=config["limit_train_batches"], 
        limit_val_batches= config["limit_val_batches"],
        max_epochs=config["max_iters"], 
        logger=wandb_logger, 
        precision=config["precision"], 
        check_val_every_n_epoch= config["eval_interval"],
        callbacks=[
            EarlyStopping(monitor="validation_loss", min_delta=config["lr"]*config["min_delta_lr_factor"], patience=2)
        ]
    )

    # train model.
    trainer.fit(model=gpt, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # test model.
    trainer.test(model=gpt, dataloaders=test_loader)
        

if __name__ == "__main__":
    # run main with default configuration.
    main()

    # Do automatic sweep.
    # # 1: Initialize wandb
    # wandb.login()    
    # # 3: Start the sweep
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="gpt")
    # wandb.agent(sweep_id, function=main, count=200)