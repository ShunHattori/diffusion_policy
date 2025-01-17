import os
import sys

# add bp_multi_goal path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.cm import get_cmap

from diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('-p', "--dataset_path", type=str, required=True)
def main(dataset_path):
    # Load replay buffer
    replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode="r")
    episode_data = replay_buffer.data["obs"]
    print(replay_buffer.get_episode_idxs())

    episode_idxs = replay_buffer.get_episode_idxs()
    episodes = {}

    for idx in np.unique(episode_idxs):
        # Get indices where episode index matches
        indices = np.where(episode_idxs == idx)[0]

        # Accumulate episode data manually
        episodes[idx] = np.array([episode_data[i] for i in indices])

    # Plot episode data
    fig, ax = plt.subplots()
    # for idx, episode in episodes.items():
    #     ax.plot(episode[:,0], episode[:,1], alpha=0.5)
    #     ax.plot(episode[:,3], episode[:,4], alpha=0.5)

    one_episode = episodes[555]

    ax.plot(one_episode[:,0], one_episode[:,1], alpha=0.5)
    ax.plot(one_episode[:,3], one_episode[:,4], alpha=0.5)
    ax.plot(one_episode[:,6], one_episode[:,7], alpha=0.5)

    ax.set_aspect('equal', 'box')

    plt.show()

if __name__ == "__main__":
    main()



'''
python dataset_2dplotter.py -p "./dataset/lowerfirst1.zarr"
'''
