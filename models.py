import torch
from torch import dropout, nn
from torch import Tensor
import numpy as np
#import scipy.io as sio
from copy import deepcopy


#########################################################################
# An Autoencoder to encode gene expression of cell-lines
#########################################################################

class Simple_AE(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 h_dims,
                 drop_out,
                 batch_norm):

        super(Simple_AE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)

        hidden_dims.insert(0, input_dim)

        # Build Encoder
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            if(batch_norm):
                modules.append(
                    nn.Sequential(
                        nn.Linear(i_dim, o_dim),
                        nn.BatchNorm1d(o_dim),
                        #nn.ReLU(),
                        nn.Dropout(drop_out))
                )
            else: 
                modules.append(
                    nn.Sequential(
                        nn.Linear(i_dim, o_dim),
                        # nn.BatchNorm1d(o_dim),
                        #nn.ReLU(),
                        nn.Dropout(drop_out))
                )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                              hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2],
                      hidden_dims[-1]),
            nn.Sigmoid()
        )
        # self.feature_extractor =nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return output, embedding

#########################################################################
# Base line model to predict synergy End to End
#########################################################################


class MyBaseline(torch.nn.Module):
    def __init__(self, data, config):

        super(MyBaseline, self).__init__()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.criterion = torch.nn.MSELoss()

        # Compute dimension of input for predictor
        predictor_layers = config["predictor_layers"]

        assert predictor_layers[-1] == 1

        self.predictor = self.get_predictor(data, config, predictor_layers)
        # self.apply(self._init_weights)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        """
        Loss function for the synergy prediction pipeline
        :param output: output of the predictor
        :param drug_drug_batch: batch of drug-drug combination examples
        :return:
        """
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


#########################################################################
# Models with uncertainty
#########################################################################
class EnsembleModel(torch.nn.Module):
    """
    Wrapper class that can handle an ensemble of models
    """

    def __init__(self, data, config):
        super(EnsembleModel, self).__init__()
        models = []
        self.ensemble_size = config["ensemble_size"]

        for _ in range(self.ensemble_size):
            models.append(MyBaseline(data, config))

        self.models = torch.nn.ModuleList(models)

    def forward(self, data, drug_drug_batch):
        comb_list = []
        for model_i in self.models:
            comb = model_i(data, drug_drug_batch)
            comb_list.append(comb)
        return torch.cat(comb_list, dim=1)

    def loss(self, output, drug_drug_batch):
        loss = 0
        for i in range(self.ensemble_size):
            output_i = output[:, i][:, None]
            loss_i = self.models[i].loss(output_i, drug_drug_batch)
            loss += loss_i

        loss /= self.ensemble_size
        return loss


class PredictiveUncertaintyModel(torch.nn.Module):
    """
    Wrapper class that can handle Predictive Uncertainty models
    """

    def __init__(self, data, config):
        super(PredictiveUncertaintyModel, self).__init__()

        self.mu_predictor = MyBaseline(data, config)
        self.std_predictor = MyBaseline(data, config)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.data = data

    def forward(self, data, drug_drug_batch):
        """
        Used to get samples from the predictive distribution. The number of samples is set to 100.
        """
        mean_prediction = self.mu_predictor(data, drug_drug_batch)
        std_prediction = self.std_predictor(data, drug_drug_batch)

        return mean_prediction + \
            torch.randn((len(mean_prediction), 100)).to(
                self.device) * torch.exp(1/2*std_prediction)

    def loss(self, output, drug_drug_batch):
        """
        the mu_predictor model is trained using MSE while the std_predictor is trained using the adaptive NLL criterion
        """
        predicted_mean = self.mu_predictor(self.data, drug_drug_batch)
        ground_truth_scores = drug_drug_batch[2][:, None]
        predicted_log_sigma2 = self.std_predictor(self.data, drug_drug_batch)

        # Loss for the mu_predictor
        MSE = (predicted_mean - ground_truth_scores) ** 2

        # Loss for the std_predictor. We cap the exponent at 80 for stability
        denom = (2 * torch.exp(torch.min(predicted_log_sigma2,
                 torch.tensor(80, dtype=torch.float32).to(self.device))))

        return torch.mean(MSE) + torch.mean(predicted_log_sigma2 / 2 + MSE.detach() / denom)


#########################################################################
# Predictor part of synergy prediction
#########################################################################


class MyMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(MyMLPPredictor, self).__init__()
        self.drug_embed_len = config['drug_embed_len']
        # TODO: if you want to use the autoencoder, this is not the best practice
        self.cell_embed_len = data.cell_line_features.shape[1]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.layer_dims = predictor_layers
        #layers on drug features
        self.drug_embed_hidden_layers = config["drug_embed_hidden_layers"]
        assert len(self.drug_embed_hidden_layers) > 0
        assert self.drug_embed_hidden_layers[-1] < data.x_drugs.shape[1]

        layers_before_cell = []
        layers_after_cell = []

        # Build early layers (for embeding drugs)
        layers_before_cell = self.add_layer(
            layers_before_cell,
            0,
            data.x_drugs.shape[1],
            self.drug_embed_hidden_layers[0],
            dropout = config['first_layer_dropout']

        )
        layers_before_cell = self.add_layer(
            layers_before_cell,
            0,
            self.drug_embed_hidden_layers[0],
            self.drug_embed_len,
            dropout = config['middle_layer_dropout']
        )

        # Build last layers (after addition of the two embeddings)
        layers_after_cell = self.add_layer(
            layers_after_cell,
            0,
            self.drug_embed_len*2+self.cell_embed_len,
            self.layer_dims[0],
            dropout=config['first_layer_dropout']
        )
        for i in range(0,len(self.layer_dims)-1):
            layers_after_cell = self.add_layer(
                layers_after_cell,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                dropout=config['middle_layer_dropout'] if i!=len(self.layer_dims)-1 else 0
            )

        self.before_merge_mlp = torch.nn.Sequential(*layers_before_cell)
        self.after_merge_mlp = torch.nn.Sequential(*layers_after_cell)
        # self.apply(self._init_weights)
 
    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)
        cell_features = data.cell_line_features[cell_lines]

        # Apply before merge MLP
        h_1 = self.before_merge_mlp(h_drug_1)
        h_2 = self.before_merge_mlp(h_drug_2)

        concatinated = torch.cat((h_1, h_2, cell_features), 1)

        comb = self.after_merge_mlp(concatinated)

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0].type('torch.LongTensor')  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1].type(
            'torch.LongTensor')  # Edge-head drugs in the batch
        # Cell line of all examples in the batch
        cell_lines = drug_drug_batch[1].type('torch.LongTensor')

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_1s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, activation=None, dropout=0):
        layers.append(nn.Linear(dim_i, dim_i_plus_1))
        if (dropout != 0):
            if i != len(self.layer_dims) - 2:
                layers.append(nn.Dropout(dropout))

        if (activation):
            layers.append(activation())
        if i != len(self.layer_dims) - 2:
            layers.append(nn.ReLU())
        return layers

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MyClassification(torch.nn.Module):
    def __init__(self, data, config):

        super(MyClassification, self).__init__()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        predictor_layers = config["predictor_layers"]
        assert predictor_layers[-1] == 1
        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        """
        Loss function for the synergy prediction pipeline
        :param output: output of the predictor
        :param drug_drug_batch: batch of drug-drug combination examples
        :return:
        """
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss
    

class WeightedRegression(torch.nn.Module):
    def __init__(self, data, config):
        super(WeightedRegression, self).__init__()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.criterion = nn.MSELoss(reduction='none')
        predictor_layers = config["predictor_layers"]
        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)
        # make a copy of targets and detach from computational graph
        weights = abs(ground_truth_scores.clone().detach())
        weights = np.log(weights.cpu() + np.e).to(self.device)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class MyMatchModel(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(MyMLPPredictor, self).__init__()
        self.cell_embed_len = data.cell_line_features.shape[1]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
    
        #constructing cell-drug-network
        self.drug_cell_layers = config["drug_cell_layers"]
        cell_drug_net = []
        cell_drug_net = self.add_layer(
            cell_drug_net,
            0,
            data.x_drugs.shape[1] + self.cell_embed_len,
            self.drug_cell_layers[0],
            dropout=config['first_layer_dropout']

        )
        cell_drug_net = self.add_layer(
            cell_drug_net,
            0,
            self.drug_cell_layers[0],
            self.drug_embed_len,
            dropout=config['middle_layer_dropout']
        )

        #synergy_net
        self.syn_layers = predictor_layers
        synergy_net = []
        synergy_net = self.add_layer(
            synergy_net,
            0,
            self.drug_cell_layers[-1]*2,
            self.syn_layers[0],
            dropout=config['first_layer_dropout']
        )
        for i in range(0, len(self.syn_layers)-1):
            synergy_net = self.add_layer(
                synergy_net,
                i,
                self.syn_layers[i],
                self.syn_layers[i + 1],
                dropout=config['middle_layer_dropout'] if i != len(
                    self.syn_layers)-1 else 0
            )
        
        #sensitivity_net
        self.sensitivity_layers = config["sensitivity_layers"]
        sen_net = []
        sen_net = self.add_layer(
            sen_net,
            0,
            self.drug_cell_layers[-1],
            self.layer_dims[0],
            dropout=config['first_layer_dropout']
        )
        for i in range(0, len(self.sensitivity_layers)-1):
            sen_net = self.add_layer(
                sen_net,
                i,
                self.sensitivity_layers[i],
                self.sensitivity_layers[i + 1],
                dropout=config['middle_layer_dropout'] if i != len(
                    self.sensitivity_layers)-1 else 0
            )

        self.drug_cell_net = torch.nn.Sequential(*cell_drug_net)
        self.synergy_net = torch.nn.Sequential(*synergy_net)
        self.sen_net = torch.nn.Sequential(*sen_net)


    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)
        cell_features = data.cell_line_features[cell_lines]

        # Apply before merge MLP
        h_1_c = self.before_merge_mlp(h_drug_1, cell_features)
        h_2_c = self.before_merge_mlp(h_drug_2, cell_features)

        concatinated = torch.cat((h_1_c, h_2_c), 1)

        comb = self.after_merge_mlp(concatinated)

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        # Cell line of all examples in the batch
        cell_lines = drug_drug_batch[1]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, activation=None, dropout=0):
        layers.append(nn.Linear(dim_i, dim_i_plus_1))
        if (dropout != 0):
            if i != len(self.layer_dims) - 2:
                layers.append(nn.Dropout(dropout))

        if (activation):
            layers.append(activation())
        if i != len(self.layer_dims) - 2:
            layers.append(nn.ReLU())
        return layers



