
from importlib import import_module
from ray import tune
from utils import get_project_root
from models import Simple_AE
from trainers import DAETrainer
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

datasets_path = "/Users/farzanehlabbaf/Documents/Drug Discovery/datasets/"

########################################################################################################################
# Configuration
########################################################################################################################

cell_autoencoder_config = {
    "name":"AE_dummy",
    "model": Simple_AE,
    "data_path": "/Users/farzanehlabbaf/Documents/Drug Discovery/models/MyProject/MyProject/data/processed/DepMap_expression_processed.csv",
    # "train_test_split": 'random',
    "use_tune": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    # "h_dims": tune.grid_search([
    #     [512],
    #     [512, 128],
    #     [1024, 512, 128]
    #     [1024, 512],
    #     [1024, 128],
    # ]),
    'h_dims': [512],
    "num_epoches": 500,
    "seed": tune.grid_search([2, 3, 4]),
    # 'seed':2,
    # Optimizer config
    # "lr": tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4]),
    'lr': 1e-2,
    "batch_size": 64,
    # "batch_size": tune.grid_search([16, 32, 64, 128]),
    "lr_step": 5e-1,
    # 'milestone': tune.grid_search([10, 20]),
    'milestone': 10,
    # "num_genes_compressed": tune.grid_search([64, 128, 256]),
    'num_genes_compressed': 128,
    # "dropout": tune.grid_search([0.1, 0.2, 0.3]),
    "dropout": 0.2,
    # "noise": tune.grid_search([0.1, 0.2, 0.3]),
    "noise": 0.2,
    "num_genes_in": 15909,
    "gene_train_loader": None,
    "gene_test_loader": None,
    "wandb_group": 'Tune_DepMap_autoencoder'
}


########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    'trainer': DAETrainer,
    'trainer_config': {
        **cell_autoencoder_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 300, "patience":10},
    # "checkpoint_score_attr": 'eval/comb_r_squared',
    # "keep_checkpoints_num": 1,
    # "checkpoint_at_end": False,
    # "checkpoint_freq": 1,
    # "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}