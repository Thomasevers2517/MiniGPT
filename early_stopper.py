from pytorch_lightning.callbacks import EarlyStopping
class CustomEarlyStoppingWithLRCheck(EarlyStopping):
    def on_epoch_end(self, trainer, pl_module):
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if current_lr < 0.001:
            print("Learning rate too low. Stopping training.")
            trainer.should_stop = True
        super().on_epoch_end(trainer, pl_module)