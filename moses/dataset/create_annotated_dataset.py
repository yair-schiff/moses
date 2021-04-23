import argparse
import time
import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm

from moses.metrics.SA_Score import sascorer


def get_parser():
    parser = argparse.ArgumentParser("Create annotated dataset with: logP, SA, and QED")
    parser.add_argument('--path_to_original', type=str, default='/dccstor/trustedgen/tda/moses_dataset/smiles',
                        help='Absolute path to original SMILES csvs directory.')
    return parser


if __name__ == '__main__':
    pars = get_parser()
    config = pars.parse_args()
    for split in ['test', 'train', 'test_scaffolds']:
        print(f'Working on split: {split}...')
        all_start = time.time()
        df = pd.read_csv(os.path.join(config.path_to_original, f'dataset_v1_{split}.csv'))

        print('\tGetting molecules from smiles...', end='\t')
        start = time.time()
        df['mol'] = df.apply(lambda row: Chem.MolFromSmiles(row['SMILES']), axis=1)
        print(f'(Took {time.time() - start:,.3f} seconds)')

        print('\tGetting logP from molecules...', end='\t')
        start = time.time()
        df['logp'] = df.apply(lambda row: Chem.Crippen.MolLogP(row['mol']), axis=1)
        print(f'(Took {time.time() - start:,.3f} seconds)')

        print('\tGetting QED from molecules...', end='\t')
        start = time.time()
        df['qed'] = df.apply(lambda row: QED.qed(row['mol']), axis=1)
        print(f'(Took {time.time() - start:,.3f} seconds)')

        print('\tGetting SA from molecules...', end='\t')
        start = time.time()
        df['sa'] = df.apply(lambda row: sascorer.calculateScore(row['mol']), axis=1)
        print(f'(Took {time.time() - start:,.3f} seconds)')

        print('\tSaving annotated csv to file...')
        df = df.drop(['mol'], axis=1)
        df.to_csv(os.path.join(config.path_to_original, f'dataset_v1_{split}_annotated.csv'))
        print(f'(Took {time.time() - all_start:,.3f} seconds)\n')
