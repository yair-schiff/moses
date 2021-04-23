import argparse
import os
import sys
import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from moses.dataset import get_dataset
from moses.metrics.SA_Score import sascorer
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Models sampler script", description="available models"
    )
    for model in MODELS.get_model_names():
        add_sample_args(subparsers.add_parser(model))
    return parser


def get_collate_fn(model, config):
    device = config.device

    def collate(data):
        data.sort(key=len, reverse=True)
        tensors = [model.string2tensor(string, device=device) for string in data]
        return tensors, data
    return collate


def get_dataloader(model, data, config, collate_fn=None, shuffle=False):
    if collate_fn is None:
        collate_fn = get_collate_fn(model, config)
    return DataLoader(data, batch_size=config.n_batch, shuffle=shuffle, num_workers=0, collate_fn=collate_fn,)


def get_embeddings(model, config, data, data_name):
    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(config.device)
    model.eval()
    data_loader = tqdm(get_dataloader(model, data, config), desc=f'{data_name.capitalize()} embeddings')
    smiles_embeddings = {}
    decoded_smiles = {}
    with torch.no_grad():
        for input_batch in data_loader:
            tensors, smiles = input_batch
            tensors = tuple(data.to(model.device) for data in tensors)
            latent_distribution = model.get_latent_distribution(tensors)
            decoded = model.sample(n_batch=latent_distribution['mu'].shape[0], z=latent_distribution['mu'])
            for i, smi in enumerate(smiles):
                smiles_embeddings[smi] = latent_distribution['mu'][i]
                decoded_smiles[smi] = decoded[i]
    return smiles_embeddings, decoded_smiles


def main(model, config):
    data_file_prefix = f'{config.model_load.split("/")[-3]}_ep{config.model_load.split("/")[-1].split("_")[-1][:-3]}'
    for data_name in ['train', 'test', 'test_scaffolds']:
        data = get_dataset(data_name)
        embeddings, decoded = get_embeddings(model, config, data, data_name)
        annotated_data = []
        for smi, embed in tqdm(embeddings.items(), desc=f'{data_name.capitalize()} annotations'):
            mol = Chem.MolFromSmiles(smi)
            qed = QED.qed(mol)
            logP = Chem.Crippen.MolLogP(mol)
            sa = sascorer.calculateScore(mol)
            annotated_data.append({'smiles': smi, 'embedding': embed.cpu(), 'decoded_smiles': decoded[smi],
                                   'qed': qed, 'logp': logP, 'sa': sa})
        torch.save(annotated_data, os.path.join(config.gen_save, f'{data_file_prefix}_moses_annotated_{data_name}.pt'))


if __name__ == "__main__":
    pars = get_parser()
    model_type = sys.argv[1]
    conf = pars.parse_args()
    main(model_type, conf)
