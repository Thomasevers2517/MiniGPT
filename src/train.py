import numpy as np
import torch
import wandb

@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters):
    losses = []
    model.eval()
    for k in range(eval_iters):
        X, Y = next(iter(dataloader))
        logits, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

from tqdm import tqdm
def train(train_loader, test_loader, model, optimizer, config):
    for i in tqdm(range(config["max_iters"])):
        if i % config["eval_interval"] == 0 or i == config["max_iters"] - 1:
            train_loss = estimate_loss(model, train_loader, config["eval_iters"])
            val_loss = estimate_loss(model, test_loader, config["eval_iters"])
            if config["wandb_logging"]:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            else:
                print(f"epoch {i}: train loss is {train_loss:.2f}, and val loss is {val_loss:.2f}")

    
        batch = next(iter(train_loader))
        x, y = batch
        logits, loss = model(x, y)
        wandb.log({"train_loss": loss})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()