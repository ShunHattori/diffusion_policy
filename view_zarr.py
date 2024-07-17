import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer

replay_buffer = ReplayBuffer.create_from_path("data/block_pushing/multimodal_push_seed.zarr", mode="a")
print(replay_buffer.get_chunks())
print(replay_buffer.data["obs"][:350])
