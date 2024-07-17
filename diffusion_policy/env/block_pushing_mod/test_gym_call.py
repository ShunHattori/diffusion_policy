import gym

# from diffusion_policy.env.block_pushing_mod import block_pushing
from diffusion_policy.env.block_pushing_mod import block_pushing_multimodal

# env = gym.make("BlockPushNormalized-v0")
env = gym.make("BlockPushMultimodal-v0")

state = env.reset()

done = False
for i in range(100):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    env.render()

    # print(f"State: {next_state}")
    # print(f"Reward: {reward}")
    # print(f"Done: {done}")
    # print(f"Info: {info}")
    state = env.get_pybullet_state()
    # print(f"State: {state}")
    # state["robots"][0] = [0, 0, 0]
    # state["robots"][1] = [0, 0, 0, 1]
    # for key, value in env.get_pybullet_state().items():
    #     print(f"{key}: {value}\n")


# 環境を終了
env.close()
