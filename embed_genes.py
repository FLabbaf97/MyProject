# import argparse
# import pickle
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pickle
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import os
# import importlib
# import pandas as pd
# from sklearn.model_selection import train_test_split



# # imports from my own codes
# from data_loaders import CellGeneDataset, collate_fn
# from trainers import train_DAE_model

# # paths
# save_path = 'saved/'
# raw_data_path = 'data/raw/'
# processed_data_path = 'data/processed/'


# def main(configuration):
#     """ Main function. """
#     opt = configuration
#     device_type = "cuda" if torch.cuda.is_available() else "cpu"
#     opt["device"] = torch.device(device_type)
#     opt['save_path'] = save_path+opt["name"]+"/"
#     if ~os.path.exists(opt['save_path']):
#         os.makedirs(opt['save_path'], exist_ok=True)

#     # prepare dataloaders for gene_expression
#     cell_features = pd.read_csv(opt['cell_AE_config']['data']).set_index('cell_line_name')
#     # Split the data into a training set and a test set
#     cell_features_train, cell_features_test = train_test_split(
#         cell_features, test_size=0.2, random_state=42)
#     cell_feature_train_dataset = CellGeneDataset(cell_features_train)
#     cell_fearute_train_dataloader = DataLoader(cell_feature_train_dataset, collate_fn=collate_fn, batch_size=opt['cell_AE_config']['batch_size'], shuffle=True)
#     cell_feature_test_dataset = CellGeneDataset(cell_features_test)
#     cell_fearute_test_dataloader = DataLoader(cell_feature_test_dataset,collate_fn=collate_fn,batch_size=opt['cell_AE_config']['batch_size'], shuffle=True)
#     opt['cell_AE_config']['gene_train_loader'], opt['cell_AE_config']['gene_test_loader'] = cell_fearute_train_dataloader, cell_fearute_test_dataloader

#     num_genes_in = cell_feature_train_dataset.cell_features.shape[1]

#     """ prepare AutoEncoder model """
#     cell_AE_config = opt['cell_AE_config']
#     AE_model = cell_AE_config['model'](
#         input_dim=num_genes_in,
#         latent_dim=cell_AE_config['num_genes_compressed'],
#         h_dims=cell_AE_config['h_dims'],
#         drop_out=cell_AE_config['dropout'],
#     )
#     AE_model.to(device_type)
#     """ optimizer and scheduler """
#     optimizer = optim.Adam(filter(lambda x: x.requires_grad, AE_model.parameters()),
#                            cell_AE_config['lr'], betas=(0.9, 0.999), eps=1e-9)
#     scheduler = optim.lr_scheduler.StepLR(
#         optimizer, 8, gamma=cell_AE_config['lr_step'])

#     """ number of parameters """
#     num_params = sum(p.numel()
#                      for p in AE_model.parameters() if p.requires_grad)
#     print('[Info] Number of parameters: {}'.format(num_params))

#     """ Train """
#     train_auto_encoder = True
#     if (train_auto_encoder):
#         model, loss_train = train_DAE_model(
#             model=AE_model,
#             data_loaders={
#                 "train": cell_AE_config['gene_train_loader'], "val": cell_AE_config['gene_test_loader']},
#             optimizer=optimizer,
#             loss_function=nn.MSELoss(),
#             n_epochs=cell_AE_config['num_epoches'],
#             scheduler=scheduler,
#             load=False,
#             config=cell_AE_config,
#             save_path=opt['save_path'],
#         )
#     else:
#         MODEL_PATH = save_path+config["save_path"]
#         checkpoint = torch.load(MODEL_PATH)
#         AE_model.load_state_dict(checkpoint)
#         AE_model.eval()

#     """" Get encoded space """
#     cell_feature_dataset = CellGeneDataset(cell_features)
#     cell_fearute_dataloader = DataLoader(
#         cell_feature_dataset, collate_fn=collate_fn, batch_size=100000, shuffle=False)
#     for batch, meta in cell_fearute_dataloader:
#         enc_out = model.encode(batch)
#         # print(dec_out.shape)
#     # print(enc_out)
#     enc_out.to_csv(save_path+'embeded_values.csv')

from utils import get_tensor_dataset, trial_dirname_creator
from ray import tune
import ray
import time
import argparse
import importlib
from trainers import ActiveTrainer, BasicTrainer
import os
import wandb
import random


os.environ['WANDB_API_KEY'] = "e0f887ce4be7bebfe48930ffcff4027f49b02425"


def train(configuration, num_cpu='all'):

    if configuration["trainer_config"]["use_tune"]:
        ###########################################
        # Use tune
        ###########################################
        if num_cpu == 'all':
            ray.init()
        else:
            ray.init(num_cpus=int(num_cpu))

        time_to_sleep = 5
        print("Sleeping for %d seconds" % time_to_sleep)
        time.sleep(time_to_sleep)
        print("Woke up.. Scheduling")

        tune.run(
            configuration["trainer"],
            name=configuration["name"],
            config=configuration["trainer_config"],
            stop=configuration["stop"],
            # resources_per_trial=configuration["resources_per_trial"],
            local_dir=configuration["summaries_dir"],
            checkpoint_freq=configuration.get("checkpoint_freq"),
            checkpoint_at_end=configuration.get("checkpoint_at_end"),
            checkpoint_score_attr=configuration.get("checkpoint_score_attr"),
            keep_checkpoints_num=configuration.get("keep_checkpoints_num"),
            trial_dirname_creator=trial_dirname_creator,
        )

    else:
        ###########################################
        # Do not use tune
        ###########################################
        trainer = configuration["trainer"](
            configuration["trainer_config"],
            )
        for i in range(configuration["trainer_config"]["num_epoch_without_tune"]):
            print("epoch ", i)
            trainer.train()


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help='Name of the configuration file without ".py" at the end',
    )
    parser.add_argument(
        "--cpu",
        type=str,
        help='number of resources',
        default='all'
    )
    args = parser.parse_args()

    # Retrieve configuration
    my_config = importlib.import_module("config." + args.config)
    print("Running with configuration from", "config." + args.config)

    # Set the name of the log directory after the name of the config file
    my_config.configuration["name"] = args.config

    # Train
    train(my_config.configuration, num_cpu=args.cpu)
