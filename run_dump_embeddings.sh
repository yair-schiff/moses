#!/bin/bash

export PYTHONPATH=":$(pwd)/moses"

source /speech7/multimodal/work/yschiff/miniconda3/bin/activate cogmol_env

jbsub \
  -proj "gf_icnn_dump" \
  -name "dump_embeddings_mol_vanilla_vae" \
  -mem 150g \
  -cores 1+1 \
  -queue x86_6h \
  -require v100 \
  -out "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/dump_embeddings_vanilla_vae.out" \
  -err "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/dump_embeddings_vanilla_vae.err" \
  python scripts/dump_embeddings.py vae \
    --gen_save "/speech7/multimodal/work/yschiff/GF-ICNN/data" \
    --model_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/checkpoints/vanilla_vae_012.pt" \
    --config_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/config.pt" \
    --vocab_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/vocab.pt" \
    --n_samples 1 \
    --n_batch 1024