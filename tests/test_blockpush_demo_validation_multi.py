import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import click
import matplotlib.pyplot as plt
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)


def print_common_data_info(replaybuffer):
    pass


def set_mpl_default(xlim=None, ylim=None, xticks_int=None, yticks_int=None, xlabel=None, ylabel=None):
    import matplotlib.ticker as ticker

    plt.style.use("ggplot")
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.gca()
    ax.axis("equal")

    if xlabel is not None:
        assert ylabel is not None, "xlabel指定時にはylabelも設定されている必要があります"
        ax.set(xlabel=xlabel, ylabel=ylabel)

    if xlim is not None:
        assert ylim is not None, "xlim指定時にはylimも設定されている必要があります"
        ax.set(xlim=xlim, ylim=ylim)

    if xticks_int is not None:
        assert yticks_int is not None, "xticks_int指定時にはyticks_intも設定されている必要があります"
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xticks_int))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks_int))

    ax.grid(which="major", lw=0.7)
    ax.grid(which="minor", lw=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    return fig, ax


def plot_and_save_episode(idx, pps, output_dir):
    fig, ax = plt.subplots()
    ax.plot(*pps, alpha=0.3)
    ax.scatter(*pps, alpha=0.3, s=1)
    ax.set(xlabel="x", ylabel="y")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.1, -0.9)

    # Save the plot
    savepath = os.path.join(output_dir, f"epi_{idx}.png")
    fig.savefig(savepath, dpi=300)
    plt.close(fig)
    print(f"Saved episode {idx} plot to {savepath}")


@click.command()
@click.option("--input", "-i", required=True, type=str, help="The path to zarr")
@click.option("--n_workers", "-n", default=5, type=int, help="Number of workers for parallel processing")
def main(input, n_workers):
    rb = ReplayBuffer.create_from_path(input, mode="a")
    print_common_data_info(rb)

    output_dir = os.path.join(os.getcwd(), "tests/test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Define the plot function with the output directory set
    plot_fn = partial(plot_and_save_episode, output_dir=output_dir)

    # Use ProcessPoolExecutor to process episodes in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for idx in range(rb.n_episodes):
            epi = rb.get_episode(idx)
            if epi is None or "robot_eef_pose" not in epi:
                print(f"Warning: Episode {idx} is missing or incomplete. Skipping.")
                continue

            robot_eef_pose = epi.get("robot_eef_pose")
            if robot_eef_pose.shape[0] == 0:
                print(f"Warning: Empty 'robot_eef_pose' data for episode {idx}. Skipping.")
                continue

            pps = np.array((robot_eef_pose[:, 0], robot_eef_pose[:, 1]))
            futures.append(executor.submit(plot_fn, idx, pps))

        # Wait for all futures to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
