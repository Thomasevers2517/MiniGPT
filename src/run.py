import torch
from torch.utils.data import DataLoader
import wandb
from tokenizer import SimpleTokenizer, OpenAITokenizer
from shakespare_dataset import ShakespareDataset
from bigram_model import BigramLanguageModel
from train import train

config = {
    "batch_size": 16, # how many independent sequences will we process in parallel?
    "block_size": 32, # what is the maximum context length for predictions?
    "max_iters": 300,
    "eval_interval": 100,
    "learning_rate": 1e-3,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "eval_iters": 200,
    "n_embd": 64, 
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.0,
    "wandb_logging": False
}

if __name__ == "__main__":
    torch.manual_seed(1337)
    
    # create wandb project. 
    if config['wandb_logging']:
        wandb.init(project="mini-GPT", config=config)
    
    # read file.
    with open('training_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # create tokenizer.
    chars = sorted(list(set(text))) 
    vocab_size = len(chars)
    tokenizer = SimpleTokenizer(chars, vocab_size)
    # tokenizer = OpenAITokenizer()

    # train and test splits
    dataset = ShakespareDataset(text, tokenizer, config["block_size"])
    n = int(0.9*len(dataset)) # first 90% will be train, rest val
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, [n, len(dataset) - n])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # create model.
    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size, config=config)
    m = model.to(config["device"])
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # train model. 
    train(train_loader, test_loader, model, optimizer, config)

    # generate new shakespare.
    context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])
    generated = m.generate(context, max_new_tokens=2000)[0].tolist()
    print(tokenizer.decode(generated))