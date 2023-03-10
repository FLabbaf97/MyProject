
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
    "name": "Farzaneh_AE_1383_15806",
    "model": Simple_AE,
    "data_path": "/Users/farzanehlabbaf/Documents/Drug Discovery/models/MyProject/MyProject/data/processed/DepMap_expression_processed_1383_15806.csv",
    # "train_test_split": 'random',
    "use_tune": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    'h_dims': [1024],
    # "h_dims":tune.grid_search([
    #     [512],
    #     [512,128],
    #     [1024],
    # ]),
    "num_epoches": 500,
    "seed": tune.grid_search([1,2,3]),
    # Optimizer config
    "lr": 1e-3,
    # "lr": tune.choice([1e-2,1e-3]),
    "batch_size": 16,
    # "batch_size": tune.choice([16,32]),
    "lr_step": 5e-1,
    'milestone': 10,
    # "num_genes_compressed": tune.grid_search([128,256]),
    "num_genes_compressed": tune.grid_search([128]),
    'dropout': 0.2,
    # "dropout": tune.grid_search([0.1,0.2,0.3]),
    # "noise": tune.choice([0, 0.1, 0.3]),
    "noise": 0,
    'batch_norm': True,
    "num_genes_in": 15806,
    "gene_train_loader": None,
    "gene_test_loader": None,
    "wandb_group": 'Save autoencoder with hdim 1024 and _1383_ cell 15806 genes'
}


########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    'trainer': DAETrainer,
    'trainer_config': {
        **cell_autoencoder_config,
        'is_wandb': True,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    # "memory": 1800,
    "stop": {"training_iteration": 500, 'patience':10,},
    "checkpoint_score_attr": 'eval/loss_mean',
    'checkpoint_score_order': "min",
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": True,
    "checkpoint_freq": 10,
    "resources_per_trial": {"gpu": 0},
    "scheduler": None,
    "search_alg": None,
}
