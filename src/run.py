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

config = {
    "batch_size": 16,
    "block_size": 32, # maximum context length for predictions?
    "limit_train_batches": 20,
    "max_iters": 100,
    "lr": 1e-3,
    "n_embd": 512, 
    "n_head": 8,
    "n_layer": 8,
    "dropout": 0.0,
}

if __name__ == "__main__":
    # torch.manual_seed(1337)
    
    # read file.
    filename = 'input.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # create tokenizer.
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
    n = int(0.9*len(dataset)) # first 90% will be train, rest val
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, [n, len(dataset) - n])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=7, 
                              persistent_workers=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # create model.
    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size, config=config)   

    # train model.
    # gpt = LitGPT(model, config)
    # wandb_logger = WandbLogger(project="gpt")
    # trainer = L.Trainer(accelerator="cpu", limit_train_batches=config["limit_train_batches"], 
    #                     max_epochs=config["max_iters"], logger=wandb_logger)
    # trainer.fit(model=gpt, train_dataloaders=train_loader)

    # # load model from checkpoint.
    ckpt_path = 'gpt/wg81oz82/checkpoints/epoch=99-step=2000.ckpt'
    checkpoint = LitGPT.load_from_checkpoint(ckpt_path, model=model, config=config) 
    model = checkpoint.model

    # generate new shakespare.
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=300)[0].tolist()
    print(tokenizer.decode(tokens=generated, special_tokens=True))