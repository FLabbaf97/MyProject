
from ray import tune
from trainers import train_epoch, eval_epoch, BasicTrainer
from utils import get_project_root
from models import MyMLPPredictor
from models import MyBaseline
from datasets.drugcomb_matrix_data import DrugCombMatrix, DrugCombMatrixWithAE
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


########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": True,
    "is_wandb": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    "seed": tune.grid_search([2, 3, 4]),
    # "seed": 2,
    # Optimizer config
    "lr": 1e-2,
    # "lr": tune.grid_search([1e-2, 1e-3]),
    "weight_decay": 1e-2,
    # "weight_decay": tune.choice([1e-1,1e-2,1e-3,1e-4]),
    "batch_size": 256,
    # "batch_size": tune.grid_search([512,256,128]),
    # 'lr_step': tune.grid_search([5e-1, 1e-1]),
    'lr_step': 5e-1,
    # 'milestones': tune.grid_search([
    #     [8, 16, 32, 64, 128, 256, 512],
    #     [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
    #         120, 130, 140, 150, 160, 170, 180, 190],
    #     [10, 30, 60, 90, 120, 150, 180, 210],
    #     [10, 20, 30, 40, 50, 70, 90, 110, 150, 190]
    # ]),
    "milestones": [10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 180, 210, 250,],
    # 'total_epoch': tune.grid_search[200,400,1000],
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch,
    "eval_epoch": eval_epoch,
    "wandb_group": 'cell_line transfer (to MCF7)'
}

predictor_config = {
    "predictor": MyMLPPredictor,
    "predictor_layers":
        [
            128,
            64,
            1,
        ],
    "predictor_layers":
    [
        128,
        64,
        1,
    ],
    "drug_embed_hidden_layers": [512, ],
    # Computation on the sum of the two drug embeddings for the last n layers
    "merge_n_layers_before_the_end": 2,
    "allow_neg_eigval": True,
    "drug_embed_len": 128,
}
autorncoder_config = {
    "data": "data/processed/DepMap_expression_processed_1383_15806.csv",
    'load_ae': True,
    'ae_path': "saved/Farzaneh_AE_1383_15806_128/DAETrainer_bca1e_00001.ae",
    'input_dim': 15806,
    'latent_dim': 128,
    'h_dims': [1024],
    'drop_out': 0.2,
}


model_config = {
    "model": MyBaseline,
    "load_model_weights": False,
}

dataset_config = {
    "dataset": DrugCombMatrixWithAE,
    "study_name": 'ALMANAC',
    "in_house_data": 'without',
    "rounds_to_include": [],
    "val_set_prop": 0.2,
    "test_set_prop": 0.1,
    "test_on_unseen_cell_line": True,
    "cell_lines_in_test":['MCF7'],
    "split_valid_train": "cell_line_level",
    "cell_line": None,  # 'PC-3',
    # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
    "target": "bliss_max",
    "fp_bits": 1024,
    "fp_radius": 2,
    'duplicate_data': True,
    "meta": False,
    'drug_one_hot': True,
    'cell_feature': 'embd_expr',
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
    "stop": {"training_iteration": 500, 'patience': 10},
    "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    # "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
    # "pruner" : tune.pruner.MedianStopper(mode='min'),
}
