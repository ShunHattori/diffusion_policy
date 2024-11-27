if __name__ == "__main__":
    import os
    import pathlib
    import shutil
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


def print_common_data_info(replaybuffer):
    pass


def set_mpl_default(fig, ax, xlim=None, ylim=None, xticks_int=None, yticks_int=None, xlabel=None, ylabel=None):
    import matplotlib.ticker as ticker

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


def set_plot_field(fig, ax):
    import matplotlib.patches as patches

    target1 = patches.Rectangle((0.2, -0.4), 0.1, 0.1, edgecolor="red", facecolor="lightcoral", zorder=99, alpha=0.3)
    target2 = patches.Rectangle((0.2, -0.7), 0.1, 0.1, edgecolor="blue", facecolor="lightblue", zorder=99, alpha=0.3)
    eef_origin = patches.Circle((-0.35, -0.5), 0.01, edgecolor="black", facecolor="gray", zorder=99, alpha=0.3)
    ax.add_patch(target1)
    ax.add_patch(target2)
    ax.add_patch(eef_origin)
    ax.text(*target1.get_center(), s="Target1", ha="center", va="center", fontsize=6, color="black", zorder=99)
    ax.text(*target2.get_center(), s="Target2", ha="center", va="center", fontsize=6, color="black", zorder=99)
    ax.text(*eef_origin.get_center(), s="start", ha="center", va="center", fontsize=6, color="black", zorder=99)


@click.command()
@click.option("--input", "-i", required=True, type=str, help="The path to zarr")
@click.option("--output", "-o", required=True, type=str, help="Directory to save plotting")
@click.option("--enable_slice", "-es", is_flag=True, default=False, type=bool, help="Whether to slice the data")
def main(input, output, enable_slice):
    if os.path.exists(output):
        if click.confirm("Directory is already exists, Are you sure to override?"):
            shutil.rmtree(output)
        else:
            return
    os.makedirs(output, exist_ok=True)

    # Load zarr data and print common information
    rb = ReplayBuffer.create_from_path(input, mode="a")
    print_common_data_info(rb)

    # setup for matplotlib
    plt.style.use("ggplot")
    fig = plt.figure()
    ax = fig.gca()
    set_mpl_default(fig, ax, xlim=(-0.5, 0.5), ylim=(-0.9, -0.1), xticks_int=0.2, yticks_int=0.2, xlabel="x", ylabel="y")
    set_plot_field(fig, ax)
    # ax.set_title("End-effector pose workspace coordinate trace")

    demo_episodes = [rb.get_episode(i) for i in range(rb.n_episodes)]
    episodes_idx = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4])
    for idx, epi in tqdm(enumerate(demo_episodes)):
        if enable_slice:
            if episodes_idx[idx] != episodes_idx[idx - 1]:
                ax.clear()
                set_mpl_default(fig, ax, xlim=(-0.5, 0.5), ylim=(-0.9, -0.1), xticks_int=0.2, yticks_int=0.2, xlabel="x", ylabel="y")
                set_plot_field(fig, ax)

        pps = np.array((epi.get("robot_eef_pose")[:, 0], epi.get("robot_eef_pose")[:, 1]))
        ax.plot(*pps, alpha=0.5, linewidth=1)
        ax.scatter(*pps, alpha=0.5, s=0.5)

        if enable_slice:
            os.makedirs(os.path.join(output, f"stage{episodes_idx[idx]}"), exist_ok=True)
            os.makedirs(os.path.join(output, f"all"), exist_ok=True)
            fig.savefig(os.path.join(output, f"stage{episodes_idx[idx]}", f"epi_{idx}.png"), dpi=300)
            fig.savefig(os.path.join(output, f"all", f"epi_{idx}.png"), dpi=300)
        else:
            fig.savefig(os.path.join(output, f"epi_{idx}.png"), dpi=300)


if __name__ == "__main__":
    main()


####
#   zarr内のデータをアンパッキングして各エピソード毎のデータ確認を行う．
#   ・各エピソードでサブステップごとの画像を特定のDIRに生成
#   ・任意のエピソードのeeの動きをトレース
#   ・全エピソードの軌跡を重ねて描画
####

"""
In ~/diffusion_policy, run command below

python tests/test_blockpush_demo_validation.py -i '/home/shun-hat/diffusion_policy/data/blockpushing_real/combined/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_data_combined'
python tests/test_blockpush_demo_validation.py -i '/home/shun-hat/diffusion_policy/data/blockpushing_real/augmented_and_combined/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_data_augmented_and_combined'

python tests/test_blockpush_demo_validation.py -es -i '/home/shun-hat/diffusion_policy/data/eval_blockpushing_real_combined/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_eval_combined'
python tests/test_blockpush_demo_validation.py -es -i '/home/shun-hat/diffusion_policy/data/eval_blockpushing_real_combined_900epoch/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_eval_combined_900epoch'
python tests/test_blockpush_demo_validation.py -es -i '/home/shun-hat/diffusion_policy/data/eval_blockpushing_real_augmented_and_combined/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_eval_augmented_and_combined'

bc1,bc2
python tests/test_blockpush_demo_validation.py -i '/home/shun-hat/diffusion_policy/data/eval_blockpushing_real_comp_non_exp/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_eval_bc1'
python tests/test_blockpush_demo_validation.py -i '/home/shun-hat/diffusion_policy/data/eval_blockpushing_real_non_expert/replay_buffer.zarr' -o '/home/shun-hat/diffusion_policy/tests/trace_eval_bc2'
"""
