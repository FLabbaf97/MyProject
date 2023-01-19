
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

from models import Simple_AE
from utils import get_project_root
import os
from ray import tune
from importlib import import_module
datasets_path = "/Users/farzanehlabbaf/Documents/Drug Discovery/datasets/"

########################################################################################################################
# Configuration
########################################################################################################################

cell_autoencoder_config = {
    "name": "AE_farzaneh",
    "model": Simple_AE,
    "data_path": "/Users/farzanehlabbaf/Documents/Drug Discovery/models/MyProject/MyProject/data/processed/DepMap_expression_processed.csv",
    # "train_test_split": 'random',
    "use_tune": True,
    # "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    'h_dims': [1024],
    # "h_dims":tune.grid_search([
    #     [512],
    #     [512,128],
    #     [1024],
    # ]),
    "num_epoches": 300,
    "seed": tune.grid_search([2, 3, 4]),
    # Optimizer config
    "lr": 1e-3,
    # "lr": tune.choice([1e-2,1e-3]),
    "batch_size": 16,
    # "batch_size": tune.choice([16,32]),
    "lr_step": 5e-1,
    'milestone': tune.choice([10]),
    # "num_genes_compressed": tune.grid_search([128,256]),
    "num_genes_compressed": 256,
    'dropout': 0.2,
    # "dropout": tune.grid_search([0.1,0.2,0.3]),
    # "noise": tune.choice([0, 0.1, 0.3]),
    "noise": 0,
    'batch_norm': tune.choice([True]),
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
    "stop": {"training_iteration": 300,},
    # "checkpoint_score_attr": 'eval/comb_r_squared',
    # "keep_checkpoints_num": 1,
    # "checkpoint_at_end": False,
    # "checkpoint_freq": 1,
    "resources_per_trial": {"gpu": 1},
    "scheduler": None,
    "search_alg": None,
}
