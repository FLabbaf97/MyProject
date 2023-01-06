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

def train(configuration):

    if configuration["trainer_config"]["use_tune"]:
        ###########################################
        # Use tune
        ###########################################
        ray.init()

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
            configuration["trainer_config"])
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
    args = parser.parse_args()

    # Retrieve configuration
    my_config = importlib.import_module("config." + args.config)
    print("Running with configuration from", "config." + args.config)

    # Set the name of the log directory after the name of the config file
    my_config.configuration["name"] = args.config

    # Train
    train(my_config.configuration)
