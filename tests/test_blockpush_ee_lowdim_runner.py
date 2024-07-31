import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.blockpush_ee_lowdim_runner import (
    BlockPushEELowdimRunner,
)


def test():
    import os

    from omegaconf import OmegaConf

    cfg_path = os.path.expanduser("~/diffusion_policy/diffusion_policy/config/task/blockpush_ee_lowdim_seed.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg["n_obs_steps"] = 1
    cfg["n_action_steps"] = 1
    cfg["past_action_visible"] = False
    runner_cfg = cfg["env_runner"]
    runner_cfg["n_train"] = 1
    runner_cfg["n_test"] = 0
    runner_cfg["obs_eef_target"] = cfg["obs_eef_target"]
    del runner_cfg["_target_"]
    dataset_cfg = cfg["dataset"]
    dataset_cfg["obs_eef_target"] = cfg["obs_eef_target"]
    runner = BlockPushEELowdimRunner(**runner_cfg, output_dir="/home/shun-hat/diffusion_policy/tmp/test")

    """
    np_obs_dict {'obs': array([[[ 0.4800007 , -0.19999935,  3.1415918 ,  0.3200007 ,
         -0.19999935,  3.1415918 ,  0.3009355 , -0.40005466,
          0.3       , -0.4       ,  0.49      ,  0.2       ,
          0.        ,  0.31      ,  0.2       ,  0.        ]]],
      dtype=float32)}

      """
    # import pdb; pdb.set_trace()

    self = runner
    env = self.env
    env.seed(seeds=self.env_seeds)
    obs = env.reset()
    runner.run()


if __name__ == "__main__":
    test()
