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
        "token": {"values": ["simple"]},
        "precision": {"values": [16]}, #can add 16 if BPE is fixed
        "eval_interval": {"values": [5]},
        "limit_val_batches": {"values": [128]},
        "min_delta_lr_factor": {"values": [60]},#{"min": 1e-2, "max": 2e-2},
    }
}