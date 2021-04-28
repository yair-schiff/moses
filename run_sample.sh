#!/bin/bash

source /speech7/multimodal/work/yschiff/miniconda3/bin/activate cogmol_env
export PYTHONPATH=":$(pwd)/moses"

jbsub \
  -proj "gf_icnn_sample" \
  -name "sample_mol_vanilla_vae_regex" \
  -mem 100g \
  -cores 1+1 \
  -queue x86_1h \
  -require v100 \
  -out "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex/sample_vanilla_vae.out" \
  -err "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex/sample_vanilla_vae.err" \
  python scripts/sample.py vae_tda \
    --model_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_regex/checkpoints/vanilla_vae_024.pt" \
    --config_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/config.pt" \
    --vocab_load "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/vocab.pt" \
    --gen_save "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae/generated_mol_ep024.csv" \
    --n_samples 30000
