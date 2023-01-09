import torch
import os
import copy
from torch.utils.data import DataLoader
from utils import get_tensor_dataset
from torch.utils.data import random_split
from ray import tune
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray.air.integrations.wandb import setup_wandb

########################################################################################################################
# Epoch loops
########################################################################################################################

def train_epoch(data, loader, model, optim, scheduler=None):

    model.train()

    epoch_loss = 0
    num_batches = len(loader)

    all_mean_preds = []
    all_targets = []

    for _, drug_drug_batch in enumerate(loader):

        out = model.forward(data, drug_drug_batch)

        # Save all predictions and targets
        all_mean_preds.extend(out.mean(dim=1).tolist())
        all_targets.extend(drug_drug_batch[2].tolist())

        loss = model.loss(out, drug_drug_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    if scheduler:
        scheduler.step()
    if (len(set(all_mean_preds)) > 2):
        epoch_comb_r_squared = stats.linregress(all_mean_preds, all_targets).rvalue**2
    else:
        epoch_comb_r_squared = -1

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared
    }
    # print("Training", summary_dict)

    return summary_dict


def eval_epoch(data, loader, model):
    model.eval()

    epoch_loss = 0
    num_batches = len(loader)

    all_out = []
    all_mean_preds = []
    all_targets = []

    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch)

            # Save all predictions and targets
            all_out.append(out)
            all_mean_preds.extend(out.mean(dim=1).tolist())
            all_targets.extend(drug_drug_batch[2].tolist())

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()

        epoch_comb_r_squared = stats.linregress(
            all_mean_preds, all_targets).rvalue**2
        epoch_spear = spearmanr(all_targets, all_mean_preds).correlation

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared,
        "spearman": epoch_spear
    }

    # print("Testing", summary_dict, '\n')

    all_out = torch.cat(all_out)

    return summary_dict, all_out


########################################################################################################################
# Basic trainer
########################################################################################################################


class BasicTrainer(tune.Trainable):
    def setup(self, config, wandb=True):
        print("Initializing regular training pipeline")
        if wandb:
            self.wandb = setup_wandb(
                config, trial_id=self.trial_id, trial_name=self.trial_name, group=config['wandb_group'])
        self.batch_size = config["batch_size"]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # device_type = 'mps'
        self.device = torch.device(device_type)
        self.training_it = 0

        # Initialize dataset
        AE_config = {}
        if config['load_ae']:
            AE_config = {'encoder_path': config['ae_path'], "input_dim": config['input_dim'],
                         "latent_dim": config['latent_dim'], "h_dims": config['h_dims'], 'drop_out': config['drop_out'], 'data':config['data']}
        dataset = config["dataset"](
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            cell_line=config["cell_line"],
            study_name=config["study_name"],
            in_house_data=config["in_house_data"],
            rounds_to_include=config["rounds_to_include"],
            AE_config=AE_config
        )

        self.data = dataset.data.to(self.device)

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        if "target" in config.keys():
            possible_target_dicts = {
                "bliss_max": self.data.ddi_edge_bliss_max,
                "bliss_av": self.data.ddi_edge_bliss_av,
                "css_av": self.data.ddi_edge_css_av,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(config)

        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Valid loader
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Initialize model
        self.model = config["model"](self.data, config)

        # Initialize model with weights from file
        load_model_weights = config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        self.model = self.model.to(self.device)
        print(self.model)

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, 10, gamma=config['lr_step'])


        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.patience = 0
        self.max_eval_r_squared = -1

    def step(self):

        eval_metrics, _ = self.eval_epoch(
            self.data, self.valid_loader, self.model)

        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
            self.scheduler,
        )

        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]
        train_metrics = [("train/" + k, v) for k, v in train_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        metrics["training_iteration"] = self.training_it
        self.training_it += 1

        # Compute patience
        if metrics['eval/comb_r_squared'] > self.max_eval_r_squared:
            self.patience = 0
            self.max_eval_r_squared = metrics['eval/comb_r_squared']
        else:
            self.patience += 1

        last_lr = self.scheduler.optimizer.param_groups[0]['lr']
        print ("last lr: ", last_lr)
        metrics['current_lr'] = last_lr
        metrics['patience'] = self.patience
        metrics['all_space_explored'] = 0
        self.wandb.log(metrics)

        return metrics

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


########################################################################################################################
# Active learning Trainer
########################################################################################################################


class ActiveTrainer(BasicTrainer):
    """
    Trainer class to perform active learning. Retrains models from scratch after each query. Uses early stopping
    """

    def setup(self, config, wandb=True):
        print("Initializing active training pipeline")
        super(ActiveTrainer, self).setup(config,wandb=False)
        if wandb:
            self.wandb = setup_wandb(
                config, trial_id=self.trial_id, trial_name=self.trial_name, group=config['wandb_group'])
        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.acquisition = config["acquisition"](config)
        self.n_epoch_between_queries = config["n_epoch_between_queries"]

        # randomly acquire data at the beginning
        self.seen_idxs = self.train_idxs[:config["n_initial"]]
        self.unseen_idxs = self.train_idxs[config["n_initial"]:]
        self.immediate_regrets = torch.empty(0)

        # Initialize variable that saves the last query
        self.last_query_idxs = self.seen_idxs

        # Initialize dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(
            self.seen_idxs, self.unseen_idxs)

        # Get the set of top 1% most synergistic combinations
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        self.best_score = scores.max()
        self.top_one_perc = set(self.unseen_idxs[torch.argsort(
            scores, descending=True)[:one_perc]].numpy())
        self.count = 0

    def step(self):
        # Check whether we have explored everything
        if len(self.unseen_loader) == 0:
            print("All space has been explored")
            return {"all_space_explored": 1, "training_iteration": self.training_it}

        # Train on seen examples
        seen_metrics = self.train_between_queries()

        # Evaluate on valid set
        eval_metrics, _ = self.eval_epoch(
            self.data, self.valid_loader, self.model)

        # Score unseen examples
        unseen_metrics, unseen_preds = self.eval_epoch(
            self.data, self.unseen_loader, self.model)

        active_scores = self.acquisition.get_scores(unseen_preds)

        # Build summary
        seen_metrics = [("seen/" + k, v) for k, v in seen_metrics.items()]
        unseen_metrics = [("unseen/" + k, v)
                          for k, v in unseen_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]

        metrics = dict(
            seen_metrics
            + unseen_metrics
            + eval_metrics
        )
        
        # Acquire new data
        print("query data")
        query = self.unseen_idxs[torch.argsort(active_scores, descending=True)[
            :self.acquire_n_at_a_time]]

        # Get the best score among unseen examples
        self.best_score = self.data.ddi_edge_response[self.unseen_idxs].max(
        ).detach().cpu()
        # remove the query from the unseen examples
        self.unseen_idxs = self.unseen_idxs[torch.argsort(
            active_scores, descending=True)[self.acquire_n_at_a_time:]]

        # Add the query to the seen examples
        self.seen_idxs = torch.cat((self.seen_idxs, query))
        metrics["seen_idxs"] = self.data.ddi_edge_idx[:, self.seen_idxs]

        # Compute proportion of top 1% synergistic drugs which have been discovered
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        metrics["top"] = self.count / len(self.top_one_perc)

        query_ground_truth = self.data.ddi_edge_response[query].detach().cpu()

        query_pred_syn = unseen_preds[torch.argsort(active_scores, descending=True)[
            :self.acquire_n_at_a_time]]
        query_pred_syn = query_pred_syn.detach().cpu()

        metrics["query_pred_syn_mean"] = query_pred_syn.mean().item()
        metrics["query_true_syn_mean"] = query_ground_truth.mean().item()

        # Diversity metric
        metrics["n_unique_drugs_in_query"] = len(
            self.data.ddi_edge_idx[:, query].unique())

        # Get the quantiles of the distribution of true synergy in the query
        for q in np.arange(0, 1.1, 0.1):
            metrics["query_pred_syn_quantile_" +
                    str(q)] = np.quantile(query_pred_syn, q)
            metrics["query_true_syn_quantile_" +
                    str(q)] = np.quantile(query_ground_truth, q)

        query_immediate_regret = torch.abs(
            self.best_score - query_ground_truth)
        self.immediate_regrets = torch.cat(
            (self.immediate_regrets, query_immediate_regret))

        metrics["med_immediate_regret"] = self.immediate_regrets.median().item()
        metrics["log10_med_immediate_regret"] = np.log10(
            metrics["med_immediate_regret"])
        metrics["min_immediate_regret"] = self.immediate_regrets.min().item()
        metrics["log10_min_immediate_regret"] = np.log10(
            metrics["min_immediate_regret"])

        # Update the dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(
            self.seen_idxs, self.unseen_idxs)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1
        self.wandb.log(metrics)

        return metrics

    def train_between_queries(self):
        # Create the train and early_stop loaders for this iteration
        iter_dataset = self.seen_loader.dataset
        train_length = int(0.8 * len(iter_dataset))
        early_stop_length = len(iter_dataset) - train_length

        train_dataset, early_stop_dataset = random_split(
            iter_dataset, [train_length, early_stop_length])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(train_dataset) > 0,
        )

        early_stop_loader = DataLoader(
            early_stop_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(early_stop_dataset) > 0,
        )

        # Reinitialize model before training
        self.model = self.config["model"](
            self.data, self.config).to(self.device)

        # Initialize model with weights from file
        load_model_weights = self.config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = self.config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        # Reinitialize optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                      weight_decay=self.config["weight_decay"])

        best_eval_r2 = float("-inf")
        patience_max = self.config["patience_max"]
        patience = 0

        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            train_metrics = self.train_epoch(
                self.data, train_loader, self.model, self.optim)

            early_stop_metrics, _ = self.eval_epoch(
                self.data, early_stop_loader, self.model)

            if early_stop_metrics["comb_r_squared"] > best_eval_r2:
                best_eval_r2 = early_stop_metrics["comb_r_squared"]
                print("best early stop r2", best_eval_r2)
                patience = 0
            else:
                patience += 1

            if patience > patience_max:
                break

        return train_metrics

    def update_loaders(self, seen_idxs, unseen_idxs):
        # Seen loader
        seen_ddi_dataset = get_tensor_dataset(self.data, seen_idxs)

        seen_loader = DataLoader(
            seen_ddi_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(seen_idxs) > 0,
        )

        # Unseen loader
        unseen_ddi_dataset = get_tensor_dataset(self.data, unseen_idxs)

        unseen_loader = DataLoader(
            unseen_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cpu"),
        )

        return seen_loader, unseen_loader

########################################################################################################################
# Gene embeding autoencoder Trainer
########################################################################################################################
def train_DAE_model(model, data_loaders={}, optimizer=None, loss_function=None, n_epochs=100, scheduler=None, load=False, config={}, save_path="", eval=False):

    if (load != False):
        if (os.path.exists(save_path)):
            model.load_state_dict(torch.load(save_path))
            return model, 0
        else:
            print("Warning: Failed to load existing file, proceed to the trainning process.")

    # dataset_sizes = {
    #     x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    phases = ['val'] if eval else ['val', 'train']
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = []

            # n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, meta, idx) in enumerate(data_loaders[phase]):

                z = x
                y = np.random.binomial(
                    1, config["noise"], (z.shape[0], z.shape[1]))
                z[np.array(y, dtype=bool), ] = 0
                x.requires_grad_(True)
                # encode and decode
                output, embeded = model(z)
                # compute loss
                loss = loss_function(output, x)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)

            # print(epoch_loss)
            if phase == 'train':
                scheduler.step()

            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch, phase] = epoch_loss
            print('{} Loss: {:.8f}. Learning rate = {}'.format(
                phase, epoch_loss, last_lr))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # Select best model wts
    if ~os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Warning: " ,save_path, " path not exist, creating path")
    torch.save(best_model_wts, save_path+'AE')
    model.load_state_dict(best_model_wts)

    return model, loss_train
