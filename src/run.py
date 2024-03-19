import torch
from torch.utils.data import DataLoader
from litmodel import LitGPT
from tokenizer import SimpleTokenizer, OpenAITokenizer
from bpe_tokenizer import BPETokenizer
from lightning.pytorch.loggers import WandbLogger
from shakespare_dataset import ShakespareDataset
from bigram_model import BigramLanguageModel
import pytorch_lightning as L
from preprocessing import get_speakers, generate_speaker_tokens, split_dialogue
import wandb
from pytorch_lightning.callbacks import EarlyStopping
sweep_configuration = {
    "method": "random", # grid, random, bayes
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [64,256]},
        "block_size": {"values": [4, 16]},
        "limit_train_batches": {"values": [128]},
        "max_iters": {"values": [20, 50, 100]},
        "lr": {"min": 1e-4, "max": 1e-2},
        "n_embd": {"values": [8, 32]},
        "n_head": {"values": [4, 8]},
        "n_layer": {"values": [4, 8]},
        "dropout": {"values": [0.0]},
        "token": {"values": [0]},
        "precision": {"values": [16]}, #can add 16 if BPE is fixed
        "eval_interval": {"values": [100]},
        "limit_val_batches": {"values": [128]},
        "min_delta": {"min": 1e-2, "max": 2e-2},
    }
}


def main():
    # torch.manual_seed(1337)
    
    wandb.init(project="gpt")
    wandb_logger = WandbLogger()
    config = wandb.config
    # read file.
    filename = 'input.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # create tokenizer.
    if config["token"]==0:
        tokenizer = SimpleTokenizer(text)
        tokens = tokenizer.encode(text)
    elif config["token"] ==1:
        tokenizer = BPETokenizer(text)
        tokens = tokenizer.encode(text)
    else:
        tokenizer = OpenAITokenizer(type='gpt2')
                # add special tokens for speakers.
        speakers = get_speakers(text)
        tokens_to_ids, speakers_to_tokens = generate_speaker_tokens(speakers, tokenizer.vocab_size)
        # extend tokenizer with new special tokens.
        tokenizer.extend(tokens_to_ids)
        # split text into dialogues with speaker tokens.
        dialogues = split_dialogue(text, speakers_to_tokens)
        # encode text.
        tokens = tokenizer.encode(dialogues)



    # train and test splits
    dataset = ShakespareDataset(tokens, config["block_size"])
    n = int(0.9*len(dataset)) # first 90% will be train, rest val and test 50/50
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n, int(0.05*len(dataset)), len(dataset) - n - int(0.05*len(dataset))])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=5, 
                              persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=5, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # create model.
    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size, config=config)   

    # train model.
    gpt = LitGPT(model, config)
    

    trainer = L.Trainer(accelerator="gpu", limit_train_batches=config["limit_train_batches"], 
                        limit_val_batches= config["limit_val_batches"],
                        max_epochs=config["max_iters"], logger=wandb_logger, 
                        precision=config["precision"], check_val_every_n_epoch= config["eval_interval"],
                        callbacks=[EarlyStopping(monitor="val_loss", min_delta= config["min_delta"], patience=4)])

    
    trainer.fit(model=gpt, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model=gpt, dataloaders=test_loader)
    # # # load model from checkpoint.
    # ckpt_path = 'gpt/wg81oz82/checkpoints/epoch=99-step=2000.ckpt'
    # checkpoint = LitGPT.load_from_checkpoint(ckpt_path, model=model, config=config) 
    # model = checkpoint.model

    # generate new shakespare.
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=300)[0].tolist()
    
    

if __name__ == "__main__":
    # 1: Initialize wandb
    wandb.login()    
    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="gpt")

    wandb.agent(sweep_id, function=main, count=3)
