import os
import sys

from diffusion_policy.common.replay_buffer import ReplayBuffer

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def concatenate():
    replay_buffer1 = ReplayBuffer.create_from_path("/home/shun-hat/diffusion_policy/data/blockpushing_real/non_expert_1/replay_buffer.zarr", "r")
    replay_buffer2 = ReplayBuffer.create_from_path("/home/shun-hat/diffusion_policy/data/blockpushing_real/bc2/replay_buffer.zarr", "r")
    combined_replay_buffer = ReplayBuffer.create_empty_zarr()

    for i in range(replay_buffer1.n_episodes):
        combined_replay_buffer.add_episode(replay_buffer1.get_episode(i))
    for i in range(replay_buffer2.n_episodes):
        combined_replay_buffer.add_episode(replay_buffer2.get_episode(i))

    combined_replay_buffer.save_to_path("/home/shun-hat/diffusion_policy/data/blockpushing_real/combined/replay_buffer.zarr")

    video_dir1 = "/home/shun-hat/diffusion_policy/data/blockpushing_real/non_expert_1/videos"
    video_dir2 = "/home/shun-hat/diffusion_policy/data/blockpushing_real/bc2/videos"
    combined_dir = "/home/shun-hat/diffusion_policy/data/blockpushing_real/combined/videos"

    os.makedirs(combined_dir, exist_ok=True)
    files1 = os.listdir(video_dir1)
    files2 = os.listdir(video_dir2)

    import shutil

    # Move files from video_dir1 to combined_dir
    for file in files1:
        shutil.copytree(os.path.join(video_dir1, file), os.path.join(combined_dir, file), dirs_exist_ok=True)

    # Move files from video_dir2 to combined_dir
    for file in files2:
        shutil.copytree(os.path.join(video_dir2, file), os.path.join(combined_dir, str(int(file) + len(files1))), dirs_exist_ok=True)


if __name__ == "__main__":
    concatenate()
