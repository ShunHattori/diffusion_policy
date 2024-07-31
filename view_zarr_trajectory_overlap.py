import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# (steps, 16)の観測データにはブロック１の位置(x,y), ブロック１の姿勢(theta), ブロック２の位置(x,y), ブロック２の姿勢(theta), エンドエフェクタの手先位置(x,y)，エンドエフェクタの目標位置(x,y)，ターゲット１の位置(x,y)，ターゲット１の姿勢(theta)，ターゲット２の位置(x,y)，ターゲット２の姿勢(theta)が含まれている
# 各エピソードに関して，２つのターゲットの位置（四角形），２つのブロックの位置（四角形），エンドエフェクタの位置（点）をアニメーションプロットするコードを生成してください．
# アニメーションはエピソード毎に初期化してください．
# また，移動する各オブジェクトの軌跡は，ステップ毎に色を濃くしてください．また，軌跡の属性に応じて使用するカラーマップの種類を変更してください．
import numpy as np
import zarr
from matplotlib.cm import get_cmap

from diffusion_policy.common.replay_buffer import ReplayBuffer

# replay_buffer = ReplayBuffer.create_from_path("data/block_pushing_mod/block_pushing_mod2.zarr", mode="a")
# replay_buffer = ReplayBuffer.create_from_path("data/block_pushing/multimodal_push_seed.zarr", mode="a")
# print(replay_buffer)
# print(replay_buffer.data["obs"][:350])


# fig, ax = plt.subplots(1, 1)
# for idx in range(replay_buffer.n_episodes):
#     plt.cla()
#     data = replay_buffer.get_episode(idx)
#     action, obs = data["action"], data["obs"]
#     print(f"Episode {idx}: {obs.shape}, {action.shape}")
#     steps = action.shape[0]
#     eepos = obs[:, 6:8]
#     # change the color of the plot for each eepos index
#     ax.scatter(eepos[:, 0], eepos[:, 1], c=np.arange(steps), cmap="viridis")
# plt.show()


def animate_episode(zarr_path: str, video_name: str, epi_slice: slice):
    fig, ax = plt.subplots()

    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
    episode_data = replay_buffer.data["obs"]

    def init():
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0.2, 0.6)
        ax.set_ylim(-0.6, 0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return []

    def update(step, step_offset, episode_ends):
        step += step_offset
        prev_step = max([value for value in episode_ends if value <= step], default=0)

        # ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0.15, 0.65)
        ax.set_ylim(-0.55, 0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        print(f"Step: {step}, Prev Step: {prev_step}")

        # データを取得
        block1_x, block1_y, block1_theta = episode_data[step, 0], episode_data[step, 1], episode_data[step, 2]
        block2_x, block2_y, block2_theta = episode_data[step, 3], episode_data[step, 4], episode_data[step, 5]
        ee_x, ee_y = episode_data[step, 6], episode_data[step, 7]
        _, _ = episode_data[step, 8], episode_data[step, 9]
        target1_x, target1_y, target1_theta = episode_data[step, 10], episode_data[step, 11], episode_data[step, 12]
        target2_x, target2_y, target2_theta = episode_data[step, 13], episode_data[step, 14], episode_data[step, 15]

        # ターゲットとブロックを描画
        target1 = patches.Rectangle(
            (target1_x - 0.02, target1_y - 0.02),
            0.08,
            0.08,
            angle=np.degrees(target1_theta),
            rotation_point="center",
            color="gray",
            alpha=0.5,
        )
        target2 = patches.Rectangle(
            (target2_x - 0.02, target2_y - 0.02),
            0.08,
            0.08,
            angle=np.degrees(target2_theta),
            rotation_point="center",
            color="gray",
            alpha=0.5,
        )
        block1 = patches.Rectangle(
            (block1_x - 0.02, block1_y - 0.02),
            0.04,
            0.04,
            angle=np.degrees(block1_theta),
            rotation_point="center",
            color="red",
        )
        block2 = patches.Rectangle(
            (block2_x - 0.02, block2_y - 0.02),
            0.04,
            0.04,
            angle=np.degrees(block2_theta),
            rotation_point="center",
            color="green",
        )

        # エンドエフェクタの位置を描画
        ee = ax.plot(ee_x, ee_y, "o", color="tab:purple")

        # オブジェクトを追加
        ax.add_patch(target1)
        ax.add_patch(target2)
        ax.add_patch(block1)
        ax.add_patch(block2)

        # 軌跡を描画
        if step > 0:
            ax.plot(episode_data[prev_step:step, 0], episode_data[prev_step:step, 1], color="tab:red")
            ax.plot(episode_data[prev_step:step, 3], episode_data[prev_step:step, 4], color="tab:green")
            ax.plot(episode_data[prev_step:step, 6], episode_data[prev_step:step, 7], color="tab:purple")

        return [target1, target2, block1, block2] + ee

    step_offset = replay_buffer.meta["episode_ends"][epi_slice.start]
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=replay_buffer.meta["episode_ends"][epi_slice.stop],
        init_func=init,
        blit=False,
        repeat=False,
        fargs=(step_offset, replay_buffer.meta["episode_ends"]),
    )
    print("Saving animation...")
    ani.save(video_name, writer="ffmpeg", fps=30, dpi=150)
    print("Animation saved!")


# animate_episode(
#     zarr_path="data/block_pushing_mod/block_pushing_mod2.zarr",
#     video_name="episode_mod_overlap.mp4",
#     epi_slice=slice(100, 120),
# )
# animate_episode(
#     zarr_path="data/block_pushing/multimodal_push_seed.zarr",
#     video_name="episode_seed_overlap.mp4",
#     epi_slice=slice(100, 120),
# )


def overlap_episode(zarr_path: str, fig_name: str, epi_slice: slice):
    fig, ax = plt.subplots()

    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
    episode_data = replay_buffer.data["obs"]

    ax.set_aspect("equal")
    ax.set_xlim(0.15, 0.65)
    ax.set_ylim(-0.55, 0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    num_episodes = epi_slice.stop - epi_slice.start
    for idx in range(epi_slice.start, epi_slice.stop):
        if idx == 0:
            step_s = 0
            step_e = replay_buffer.meta["episode_ends"][idx]
        else:
            step_s = replay_buffer.meta["episode_ends"][idx - 1]
            step_e = replay_buffer.meta["episode_ends"][idx]
        ee_x, ee_y = episode_data[step_s:step_e, 6], episode_data[step_s:step_e, 7]
        ax.plot(
            ee_x,
            ee_y,
            "o",
            color=get_cmap("viridis")((idx - epi_slice.start) / num_episodes),
            alpha=0.1,
            markersize=0.2,
        )

    plt.savefig(fig_name, dpi=300, bbox_inches="tight")


# overlap_episode(
#     zarr_path="data/block_pushing_mod/block_pushing_mod2.zarr",
#     fig_name="episode_mod_overlap.png",
#     epi_slice=slice(0, 999),
# )

# overlap_episode(
#     zarr_path="data/block_pushing/multimodal_push_seed.zarr",
#     fig_name="episode_seed_overlap.png",
#     epi_slice=slice(0, 999),
# )

# overlap_episode(
#     zarr_path="data/block_pushing3.zarr",
#     fig_name="episode_block_pushing3.png",
#     epi_slice=slice(0, 999),
# )


# overlap_episode(
#     zarr_path="data/block_pushing_mod/block_pushing_mod_seed.zarr",
#     fig_name="episode_block_pushing_mod_seed_fixed.png",
#     epi_slice=slice(0, 999),
# )
