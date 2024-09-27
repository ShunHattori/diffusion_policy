import gym
from gym.wrappers import FlattenObservation

from diffusion_policy.env.block_pushing_mod import (
    block_pushing,
    block_pushing_multimodal,
)
from diffusion_policy.env.block_pushing_mod.block_pushing_multimodal import (
    BlockPushMultimodal,
)
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

env = FlattenObservation(
    BlockPushMultimodal(
        control_frequency=10,
        # image_size=(
        #     block_pushing.IMAGE_HEIGHT,
        #     block_pushing.IMAGE_WIDTH,
        # ),  # 画像のサイズはblock_pushingで定義
        shared_memory=False,
        seed=42,
        abs_action=False,
    )
)

state = env.reset()

done = False
for i in range(1):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    # print(f"{env.observation(next_state) =}")
    env.render()

    print(f"State: {next_state}")
    # print(f"Reward: {reward}")
    # print(f"Done: {done}")
    # print(f"Info: {info}")
    # state = env.get_pybullet_state()
    # print(f"State: {state}")
    # state["robots"][0] = [0, 0, 0]
    # state["robots"][1] = [0, 0, 0, 1]
    # for key, value in env.get_pybullet_state().items():
    #     print(f"{key}: {value}\n")


# 環境を終了
env.close()
