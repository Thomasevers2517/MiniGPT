import torch
def get_batch(data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y