import pytorch_lightning as L
import torch
from torch.nn import functional as F

class LitGPT(L.LightningModule):
    def __init__(self, model, config):
        super(LitGPT, self).__init__()
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)  

        # compute loss
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = y.view(B*T)
        loss = F.cross_entropy(logits, targets)

        # logs average across the epoch, to the progress bar and logger.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"])
        return optimizer