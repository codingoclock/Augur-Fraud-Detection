# config.py — these are the values to use

# Class mapping for elliptic_txs_classes.csv: 1=fraud, 0=licit, -1=unlabelled.
# This {1, 0, -1} convention is used EVERYWHERE in this codebase except
# training/loss.py's SimilarityAuxLoss, which needs its own {-1, +1} signed
# remapping (paper Eq. 4's y_v in {-1, +1}) for labelled fraud/licit nodes
# only -- do not let the two conventions collide; SimilarityAuxLoss's caller
# is responsible for excluding unlabelled (-1 in THIS convention) nodes via
# a mask before remapping the rest to {-1, +1}.
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}

CARE_GNN_CONFIG = {
    "feature_dim": 165,      # Elliptic: 167 raw cols - txId - time_step = 165 features
    "hidden_dim": 64,
    "num_classes": 2,
    "num_relations": 3,      # tdt, tbt, tft
    "num_layers": 2,
    "lr": 0.01,
    "weight_decay": 1e-4,    # covers Eq. 11's lambda2 * ||Theta||^2 via the optimizer,
                              # not computed manually in the loss (double-regularizing
                              # both would be wrong -- pick one, this is the simpler one)
    "epochs": 100,
    "batch_size": 512,       # NOTE (v3): this project trains full-graph per epoch, not
                              # mini-batch -- a documented deviation from the paper's
                              # mini-batch design, justified by Elliptic's much smaller
                              # scale vs. the paper's benchmarks and no GPU in this
                              # environment. Revisited at Level 6.
    "dropout": 0.5,
    "seed": 42,
    "train_time_steps": range(1, 35),   # steps 1-34
    "test_time_steps": range(35, 50),   # steps 35-49

    # RL selector config (v3 -- replaces v2's step-decay params entirely)
    "selector_init_p": 0.5,
    "tau": 0.05,                          # paper: fixed small step, no decay
    "rl_terminal_window": 10,             # Eq. 7's "e-10" lookback; the actual summed
                                            # window is terminal_window + 1 = 11 terms
                                            # (e-10 through e, inclusive) -- see selector.py
    "rl_terminal_reward_threshold": 2.0,  # Eq. 7: |sum of the 11-term window| <= this -> freeze

    # Auxiliary similarity loss (Eq. 11), layer-1 only
    "lambda1": 1.0,  # not given an exact value in the paper excerpt available here --
                      # treated as a tunable hyperparameter; report whatever value is
                      # actually used in the README rather than presenting 1.0 as if
                      # it were paper-specified
}

# Class imbalance handling
FRAUD_CLASS_WEIGHT = 9.0  # aggregate ratio; verify per-split ratios before training
                            # (train ~11.6% vs test ~6.5% fraud among labelled nodes,
                            # per Level 2's real graph stats)
