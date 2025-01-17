"""
in dir of
cd /home/shun-hat/diffusion_policy
python dev_policy_comp/bp_multi_goal/eval_composition.py -c1 checkpoint1 -c2 checkpoint2 -o dev_policy_comp/output
python dev_policy_comp/bp_multi_goal/eval_composition.py -c1 data/outputs/2025.01.06/18.29.57_train_diffusion_transformer_lowdim_blockpush_lowdim_mod_seed/checkpoints/latest.ckpt -c2 data/outputs/2025.01.06/18.30.41_train_diffusion_transformer_lowdim_blockpush_lowdim_mod_seed/checkpoints/latest.ckpt -o dev_policy_comp/bp_multi_goal/output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib

import click
import dill
import hydra
import torch
import wandb

from dev_policy_comp.bp_multi_goal.diffusion_transformer_lowdim_composition import (
    CompositionPolicy,
    CompositionPolicyMultiNormalizer,
)
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option("-c1", "--checkpoint1", required=True)
@click.option("-c2", "--checkpoint2", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint1, checkpoint2, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    # コンフィグは二個，wsは１つでok ws関係ないやん
    payload1 = torch.load(open(checkpoint1, "rb"), pickle_module=dill)
    payload2 = torch.load(open(checkpoint2, "rb"), pickle_module=dill)
    cfg1 = payload1["cfg"]
    cfg2 = payload2["cfg"]
    cls1 = hydra.utils.get_class(cfg1._target_)
    cls2 = hydra.utils.get_class(cfg2._target_)
    workspace1 = cls1(cfg1, output_dir=output_dir)
    workspace2 = cls2(cfg2, output_dir=output_dir)
    workspace1: BaseWorkspace
    workspace2: BaseWorkspace
    workspace1.load_payload(payload1, exclude_keys=None, include_keys=None)
    workspace2.load_payload(payload2, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy1 = workspace1.model
    if cfg1.training.use_ema:
        policy1 = workspace1.ema_model
    policy2 = workspace2.model
    if cfg2.training.use_ema:
        policy2 = workspace2.ema_model

    # policyは自作したCompを直接呼び出す．
    # Comp_policyは2つの異なるUnetPolicyを持っていて，predict_actionも2つの異なるconditional_samplingから構成される．
    # env_runnerにはComp_policyを渡す．runner内部でpredict_action，conditional_samplingが呼び出される．

    # get policy from workspace
    # ベースのポリシーがTransformerなので，ベースポリシーは対応したものを使用しなければならない．
    policy = CompositionPolicyMultiNormalizer(
        model1=policy1.model,
        model2=policy2.model,
        noise_scheduler=hydra.utils.instantiate(cfg1.policy.noise_scheduler),
        normalizer1=policy1.normalizer,
        normalizer2=policy2.normalizer,
        horizon=cfg1.policy.horizon,
        obs_dim=cfg1.policy.obs_dim,
        action_dim=cfg1.policy.action_dim,
        n_action_steps=cfg1.policy.n_action_steps,
        n_obs_steps=cfg1.policy.n_obs_steps,
        num_inference_steps=cfg1.policy.num_inference_steps,
        obs_as_cond=cfg1.policy.obs_as_cond,
        pred_action_steps_only=cfg1.pred_action_steps_only,
    )

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    env_runner = hydra.utils.instantiate(cfg1.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
