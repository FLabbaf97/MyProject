import torch
from torch.utils.data import Dataset
import numpy as np

# threshholds
syn_threshold = 30
ri_threshold = 50


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    out , meta, idx = list(zip(*insts))
    out = torch.tensor(np.array(out)).float()

    return out , meta, idx


class CellGeneDataset(Dataset):
    def __init__(self, cell_features):
        self.cell_features = cell_features.drop(columns=['cancer_type','disease','tissue'])
        self.meta = cell_features[['cancer_type','disease','tissue']]

    def __len__(self):
        return len(self.cell_features)

    def __getitem__(self, idx):
        genes = self.cell_features.iloc[idx]
        meta = self.meta.iloc[idx]
        return genes,meta,idx


# class DrugCombDataset(Dataset):
#     def __init__(self, df, drug_features, cell_features):
#         self.df = df[['drug_row', 'drug_col', 'cell_line',
#                       'ri_row', 'ri_col', 'loewe_synrgy']]
#         self.meta = df
#         self.drug_features = drug_features
#         self.cell_features = cell_features
#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

#         d1 = self.df.iloc[idx, 0]
#         d2 = self.df.iloc[idx, 1]
#         cell = self.df.iloc[idx, 2]
#         ri_d1 = 1.0 if self.df.iloc[idx, 3] > ri_threshold else 0
#         ri_d2 = 1.0 if self.df.iloc[idx, 4] > ri_threshold else 0
#         syn = 1.0 if self.df.iloc[idx, 5] > syn_threshold else 0

#         #external features
#         d1_fp = np.array(self.drug_features.loc[d1, 'fps'])
#         # d1_sm = self.drug_features.loc[d1, 'smiles']
#         # d1_sm = np.pad(d1_sm, pad_width=(0, max_drug_sm_len -
#                     #    len(d1_sm)), mode='constant', constant_values=0)
#         # d1_gn = drugGeneCompressed[d1]

#         d2_fp = np.array(self.drug_features.loc[d2, 'fps'])
#         # d2_sm = self.drug_features.loc[d2, 'smiles']
#         # d2_sm = np.pad(d2_sm, pad_width=(0, max_drug_sm_len -
#                     #    len(d2_sm)), mode='constant', constant_values=0)
#         # d2_gn = drugGeneCompressed[d2]

#         c_ts = self.cell_features.loc[cell, 'tissue']
#         c_ds = self.cell_features.loc[cell, 'disease']
#         c_gn = self.cell_features.loc[cell, :].drop(columns=['tissue','disease'])

#         sample = {
#             'd1': d1, # index of drug 1
#             'd1_fp': d1_fp,
#             # 'd1_sm': d1_sm,
#             # 'd1_gn': d1_gn,

#             'd2': d2,
#             'd2_fp': d2_fp,
#             # 'd2_sm': d2_sm,
#             # 'd2_gn': d2_gn,

#             'cell': cell,
#             'c_ts': c_ts,
#             'c_ds': c_ds,  # missing -1
#             'c_gn': c_gn,

#             'ri_d1': ri_d1,
#             'ri_d2': ri_d2,
#             'syn': syn
#         }

#         return sample



