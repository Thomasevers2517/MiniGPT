from torch.utils.data import Dataset
import torch 

class ShakespareDataset(Dataset):
    def __init__(self, text, tokenizer, block_size, device):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        x, y = x.to(self.device), y.to(self.device) # move to device for speed
        return x, y