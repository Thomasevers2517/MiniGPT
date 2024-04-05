import torch
from torch.utils.data import DataLoader
from model.litmodel import LitGPT
from tokenizer.simple import SimpleTokenizer
from tokenizer.openai import OpenAITokenizer
from tokenizer.bpe import BPETokenizer
from lightning.pytorch.loggers import WandbLogger
from dataset import ShakespareDataset
from model.bigram import BigramLanguageModel
import pytorch_lightning as L
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
        "token": {"values": [0]},
        "precision": {"values": [16]}, #can add 16 if BPE is fixed
        "eval_interval": {"values": [5]},
        "limit_val_batches": {"values": [128]},
        "min_delta_lr_factor": {"values": [60]},#{"min": 1e-2, "max": 2e-2},
    }
}

config = {
    "batch_size": 2,
    "limit_train_batches": 5,
    "max_iters": 3,
    "lr": 1e-4,
    "block_size": 16,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
    "dropout": 0.0,
    "token": "simple",
    "precision": 16, 
    "eval_interval": 20,
    "limit_val_batches": 5,
    "min_delta_lr_factor": 5
}

def main():
    torch.manual_seed(1337)

    # log to wandb.
    wandb.init(project="test", config=config)
    wandb_logger = WandbLogger()
    # config = wandb.config
    
    # read file.
    filename = 'input.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # determine tokenizer.
    if config["token"] == "simple":
        tokenizer = SimpleTokenizer(text)
    if config["token"] == "bpe":
        tokenizer = BPETokenizer(text)
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
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=7, 
                              pin_memory=True, persistent_workers=True, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=7, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                             shuffle=False, num_workers=7)

    # create model.
    gpt = LitGPT(vocab_size=tokenizer.vocab_size, **config)
    
    # create trainer.
    trainer = L.Trainer(
        accelerator="cpu", 
        limit_train_batches=config["limit_train_batches"], 
        limit_val_batches= config["limit_val_batches"],
        max_epochs=config["max_iters"], 
        logger=wandb_logger, 
        log_every_n_steps=1,
        # precision=config["precision"], 
        check_val_every_n_epoch= config["eval_interval"],
        callbacks=[ModelCheckpoint(every_n_epochs=1, verbose=True)]
        # callbacks=[EarlyStopping(monitor="validation_loss", min_delta=config["lr"]*config["min_delta_lr_factor"], patience=2)]
    )

    # train model.
    trainer.fit(model=gpt, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # test model.
    # trainer.test(model=gpt, dataloaders=test_loader)
        

if __name__ == "__main__":
    # Do automatic sweep.
    # # 1: Initialize wandb
    # wandb.login()    
    # # 3: Start the sweep
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="gpt")

    # wandb.agent(sweep_id, function=main, count=200)

    main()