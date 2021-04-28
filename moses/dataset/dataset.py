import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']
AVAILABLE_ANNOTATIONS = ['logp', 'qed', 'sa']


class AnnotatedMolecules(Dataset):
    def __init__(self, path_to_csv, annotations):
        self._path_to_csv = path_to_csv
        df = pd.read_csv(path_to_csv)
        self._data = {'smiles': df['SMILES'].tolist()}
        self._annotations = annotations
        for anno in annotations:
            assert anno in AVAILABLE_ANNOTATIONS, f'Invalid annotation provided. Only {AVAILABLE_ANNOTATIONS} allowed.'
            self._data[anno] = df[anno].tolist()

    def __getitem__(self, index):
        return_item = {'smiles': self._data['smiles'][index]}
        for anno in self._annotations:
            return_item[anno] = self._data[anno][index]
        return return_item

    def __len__(self):
        return len(self._data['smiles'])

    @property
    def data(self):
        return self._data


def get_dataset(split='train'):
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(__file__)
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, 'data', split+'.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def get_statistics(split='test'):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'data', split+'_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()
