#!/bin/bash

export PYTHONPATH=":$(pwd)/moses"

source /speech7/multimodal/work/yschiff/miniconda3/bin/activate cogmol_env

jbsub \
  -proj "gf_icnn_dump" \
  -name "dump_embeddings_mol_vanilla_vae" \
  -mem 150g \
  -cores 1+1 \
  -queue x86_24h \
  -require v100 \
  -out "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex_klw0.009_epoch180/dump_embeddings_vanilla_vae.out" \
  -err "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex_klw0.009_epoch180/dump_embeddings_vanilla_vae.err" \
  python scripts/dump_embeddings.py vae \
    --gen_save "/speech7/multimodal/work/yschiff/GF-ICNN/data" \
    --model_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex_klw0.009_epoch180/checkpoints/vanilla_vae_180.pt" \
    --config_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex_klw0.009_epoch180/config.pt" \
    --vocab_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex_klw0.009_epoch180/vocab.pt" \
    --n_samples 1 \
    --n_batch 1024 \
    --decoding "greedy" \
    --seed 0