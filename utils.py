from pathlib import Path
#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
from torch.utils.data import TensorDataset


def get_project_root():
    return Path(__file__).parent


def get_fingerprint(smile, radius, n_bits):
    if smile == "none":
        return np.array([-1] * n_bits)
 #   try:
  #      return np.array(
   #         AllChem.GetMorganFingerprintAsBitVect(
    #            Chem.MolFromSmiles(smile), radius, n_bits
     #       )
      #  )
  #  except Exception as ex:
   #     return np.array([-1] * n_bits)


########################################################################################################################
# Get TensorDataset
########################################################################################################################


def get_tensor_dataset(data, idxs):

    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        data.ddi_edge_response[idxs],
    )

########################################################################################################################
# Ray Tune
########################################################################################################################


def trial_dirname_creator(trial):
    return str(trial)

########################################################################################################################
# Map name to ID
########################################################################################################################

class Mapping():

    def __init__(self, items):
        self.item2idx = {}
        self.idx2item = []

        for idx, item in enumerate(items):
            self.item2idx[item] = idx
            self.idx2item.append(item)

    def add(self, item):
        if item not in self.idx2item:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item)-1

########################################################################################################################
# Map cell-line names
########################################################################################################################


def find_cell_lines(cell_list, cell_df):
    name_dict = {}
    names = cell_df.index
    for name in cell_list:
        if name in names:
            name_dict[name]=name
        elif name.replace('-',"") in names:
            name_dict[name.replace('-', "")] = name
        elif name.replace(' ', "") in names:
            name_dict[name.replace(' ', "")] = name
        elif name.replace(' ', "_") in names:
            name_dict[name.replace(' ', "_")] = name
        elif name.replace(' ', "-") in names:
            name_dict[name.replace(' ', "-")] = name
        elif name.replace('-', "_") in names:
            name_dict[name.replace('-', "_")] = name
        elif name.replace('_', "") in names:
            name_dict[name.replace('_', "")] = name
        elif name.replace('_', "-") in names:
            name_dict[name.replace('_', "-")] = name
        elif name.replace('/', "-") in names:
            name_dict[name.replace('/', "-")] = name
        elif name.replace('/', "") in names:
            name_dict[name.replace('/', "")] = name
        elif name.replace('/', " ") in names:
            name_dict[name.replace('/', " ")] = name
        elif name.replace('/', "_") in names:
            name_dict[name.replace('/', "_")] = name
        elif name.replace('(', "").replace(")","") in names:
            name_dict[name.replace('(', "").replace(")","")] = name
        elif name.replace('(', "_").replace(")", "_") in names:
            name_dict[name.replace('(', "_").replace(")", "_")] = name
        elif name.replace('(', "_").replace(")", "") in names:
            name_dict[name.replace('(', "_").replace(")", "")] = name
        else:
            print(" I couldn't find this damn fucking name ", name)
    for name_prime in names:
        if name_prime.replace('-', "") in cell_list:
            name_dict[name_prime] = name_prime.replace('-', "")
        elif name_prime.replace(' ', "") in cell_list:
            name_dict[name_prime] = name_prime.replace(' ', "")
        elif name_prime.replace(' ', "_") in cell_list:
            name_dict[name_prime] = name_prime.replace(' ', "_")
        elif name_prime.replace(' ', "-") in cell_list:
            name_dict[name_prime] = name_prime.replace(' ', "-")
        elif name_prime.replace('_', "") in cell_list:
            name_dict[name_prime] = name_prime.replace('_', "")
        elif name_prime.replace('/', "") in cell_list:
            name_dict[name_prime] = name_prime.replace('/', "")
        elif name_prime.replace('/', " ") in cell_list:
            name_dict[name_prime] = name_prime.replace('/', " ")
        elif name_prime.replace('(', "").replace(")", "") in cell_list:
            name_dict[name_prime] = name_prime.replace(
                '(', "").replace(")", "")
        elif name_prime.replace('(', "_").replace(")", "_") in cell_list:
            name_dict[name_prime] = name_prime.replace(
                '(', "_").replace(")", "_")
        elif name_prime.replace('(', "_").replace(")", "") in cell_list:
            name_dict[name_prime] = name_prime.replace(
                '(', "_").replace(")", "")
        # else:
        #     print(" I couldn't find this damn fucking name ", name_prime)
    return name_dict
