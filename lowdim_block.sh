#!/bin/bash

source /work/shun-hat/miniforge3/bin/activate

source /work/shun-hat/.bashrc
eval "$(conda shell.bash hook)"

conda activate robodiff
cd diffusion_policy/
python train.py --config-dir=. --config-name=low_dim_block_pushing_mod_diffusion_policy_transformer.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
