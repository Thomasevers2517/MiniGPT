import pytorch_lightning as L
import torch
from torch.nn import functional as F
from model.bigram import BigramLanguageModel


class LitGPT(L.LightningModule):

    def __init__(
        self,
        vocab_size=None, 
        batch_size=256, 
        limit_train_batches=64, 
        max_iters=400, 
        lr=1e-5, 
        block_size=16, 
        n_embd=32, 
        n_head=4, 
        n_layer=4, 
        dropout=0.0, 
        token="simple", 
        precision=16, 
        eval_interval=5, 
        limit_val_batches=128, 
        min_delta_lr_factor=60
    ):
        super(LitGPT, self).__init__()
        
        self.save_hyperparameters()
        
        # create model
        self.model = BigramLanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout
        )

    def _compute_loss(self, batch):
        x, y = batch
        logits = self.model.forward(x)  
        # compute loss
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = y.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        # log average loss across epoch to the progress bar and logger.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        loss = self._compute_loss(batch)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx) :
        loss = self._compute_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer