#!/bin/bash

export PYTHONPATH=":$(pwd)/moses"

source /speech7/multimodal/work/yschiff/miniconda3/bin/activate cogmol_env

jbsub \
  -proj "gf_icnn_train" \
  -name "train_mol_vanilla_vae_epoch1k" \
  -mem 150g \
  -cores 1+1 \
  -queue x86_24h \
  -require v100 \
  -out "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/train_vanilla_vae.out" \
  -err "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/train_vanilla_vae.err" \
  python scripts/train.py vae \
    --model_save "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/checkpoints/vanilla_vae.pt" \
    --config_save "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/config.pt" \
    --log_file "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/train_log.csv" \
    --vocab_save "/speech7/multimodal/work/yschiff/GF-ICNN/models/molecule_vae_epoch1k/vocab.pt" \
    --kl_w_end 0.05 \
    --save_frequency 20 \
    --n_batch 1024 \
    --lr_n_period 100
