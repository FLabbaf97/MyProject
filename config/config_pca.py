
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

from datasets.drugcomb_matrix_data import DrugCombMatrix, DrugCombMatrixWithAE
from models import MyBaseline
from models import MyMLPPredictor
from utils import get_project_root
from trainers import train_epoch, eval_epoch, BasicTrainer
import os


########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": False,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    # "seed": tune.grid_search([2, 3, 4]),
    "seed": 2,
    # Optimizer config
    "lr": 1e-2,
    "weight_decay": 1e-1,
    "batch_size": 1024,
    'lr_step': 5e-1,
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch,
    "eval_epoch": eval_epoch,
}

predictor_config = {
    "predictor": MyMLPPredictor,
    "predictor_layers":
        [
            128,
            64,
            1,
        ],
    "drug_embed_hidden_layers":
        [
            512,
        ],
    # Computation on the sum of the two drug embeddings for the last n layers
    "merge_n_layers_before_the_end": 2,
    "allow_neg_eigval": True,
    "drug_embed_len": 128,
}
autorncoder_config = {
    "data": "data/processed/DepMap_expression_processed.csv",
    'load_ae': False,
    'ae_path': 'saved/depMap_config/AE',
    'input_dim': 15909,
    'latent_dim': 128,
    'h_dims': [512],
    'drop_out': 0.2,
}


model_config = {
    "model": MyBaseline,
    "load_model_weights": False,
}

dataset_config = {
    "dataset": DrugCombMatrix,
    "study_name": 'ALMANAC',
    "drug_one_hot": True,
    "in_house_data": 'without',
    "rounds_to_include": [],
    "val_set_prop": 0.2,
    "test_set_prop": 0.1,
    "test_on_unseen_cell_line": False,
    "split_valid_train": "pair_level",
    "cell_line": None,  # 'PC-3',
    # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
    "target": "bliss_max",
    "fp_bits": 1024,
    "fp_radius": 2,
    'duplicate_data': False,
    'cell_feature': 'pca',

}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": BasicTrainer,  # PUT NUM GPU BACK TO 1
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
        **autorncoder_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 1000, 'patience': 10},
    "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}
