import matplotlib.pyplot as plt
import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer

# ax, fig = plt.subplots(2, 1)
# ax1 = fig[0]
# ax2 = fig[1]


# replay_buffer = ReplayBuffer.create_from_path("data/block_pushing_mod/block_pushing_mod2.zarr", mode="a")
# replay_buffer.set_chunks({"obs": replay_buffer.data["obs"].shape, "action": replay_buffer.data["action"].shape})
# print(replay_buffer)
# print(replay_buffer.get_chunks())
# print(replay_buffer.meta["episode_ends"][:])
# x = replay_buffer.data["obs"][:1000].T
# ax1.scatter(replay_buffer.meta["episode_ends"][:], range(replay_buffer.meta["episode_ends"].shape[0]))
# ax1.plot(replay_buffer.data["action"][:1000].T)

# replay_buffer = ReplayBuffer.create_from_path("data/block_pushing_mod/block_pushing_mod2.zarr", mode="a")
# replay_buffer = ReplayBuffer.create_from_path("data/block_pushing/multimodal_push_seed.zarr", mode="a")
# print(replay_buffer)
# print(replay_buffer.get_chunks())
# print(replay_buffer.meta["episode_ends"][:])
# x = replay_buffer.data["obs"][:1000].T
# ax2.scatter(replay_buffer.meta["episode_ends"][:], range(replay_buffer.meta["episode_ends"].shape[0]))
# ax2.plot(replay_buffer.data["action"][:1000].T)
# plt.show()


# # LowdimDatasetの作法に則ってデータを取り出して見たが，chuckingは関係ないことを確認．
# replay_buffer = ReplayBuffer.copy_from_path("data/pusht/pusht_cchi_v7_replay.zarr", keys=["keypoint", "action"])
# print(replay_buffer["keypoint"].shape)
# print(replay_buffer["action"].shape)
# # print(replay_buffer.get_chunks())

replay_buffer = ReplayBuffer.create_from_path("data/block_pushing_ee.zarr", mode="a")
replay_buffer.set_chunks({"obs": replay_buffer.data["obs"].shape, "action": replay_buffer.data["action"].shape})
print(replay_buffer)
print(replay_buffer.get_chunks())
