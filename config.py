# config.py — these are the values to use

# Class mapping for elliptic_txs_classes.csv: 1=fraud, 0=licit, -1=unlabelled
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}

CARE_GNN_CONFIG = {
    "feature_dim": 165,      # Elliptic: 166 cols minus txid = 165 features
    "hidden_dim": 64,
    "num_classes": 2,
    "num_relations": 3,      # tdt, tbt, tft
    "num_layers": 2,
    "lr": 0.01,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 512,       # node batch size for mini-batch training
    "dropout": 0.5,
    "seed": 42,
    "train_time_steps": range(1, 35),   # steps 1-34
    "test_time_steps": range(35, 50),   # steps 35-49

    # RL selector config (v2)
    "selector_init_p": 0.5,
    "selector_init_step": 0.02,
    "selector_min_step": 0.001,
    "selector_step_decay": 0.985,
    "selector_p_min": 0.05,
    "selector_p_max": 1.0,
}

# Class imbalance handling
FRAUD_CLASS_WEIGHT = 9.0  # ~9:1 ratio of licit:illicit in labelled nodes
