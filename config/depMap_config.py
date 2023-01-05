

from MyProject.models import Simple_AE
from MyProject.utils import get_project_root
# from MyProject.Main import gene_expression_path, comb_path, drug_path
import os
from ray import tune
from importlib import import_module
datasets_path = "/Users/farzanehlabbaf/Documents/Drug Discovery/datasets/"

########################################################################################################################
# Configuration
########################################################################################################################

cell_autoencoder_config = {
    "model": Simple_AE,
    "data": "data/processed/DepMap_expression_processed.csv",
    "train_test_split": 'random',
    "use_tune": True,
    # "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    "num_epoches": 500,
    # "seed": tune.grid_search([2, 3, 4]),
    # Optimizer config
    # "lr": tune.loguniform(1e-4, 1e-1),
    "lr": 1e-3,
    "batch_size": 64,
    # "batch_size": tune.choice([8, 16, 32, 64, 128]),
    "lr_step": 5e-1,
    "num_genes_compressed": 128,
    "dropout": 0.2,
    "noise": 0,
    "num_genes_in": 15909,
    "gene_train_loader": None,
    "gene_test_loader": None,

    # Train epoch and eval_epoch to use
    # "train_epoch": train_epoch,
    # "eval_epoch": eval_epoch,
    # "AE_layers":
    #     [
    #         2048,
    #         128,
    #         64,
    #         1,
    #     ],
}


########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "cell_AE_config": cell_autoencoder_config,
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 2, 'patience': 2},
    # "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}
