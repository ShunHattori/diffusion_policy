import gym

"""Oracle for multimodal pushing task."""
import random
from collections import OrderedDict

import click
import numpy as np

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import
import pygame

import diffusion_policy.env.block_pushing.oracles.pushing_info as pushing_info_module
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.block_pushing import block_pushing_multimodal
from diffusion_policy.env.block_pushing.utils.utils_pybullet import ObjState, XarmState
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


class OrientedPushOracle:
    """Oracle for pushing task which orients the block then pushes it."""

    def __init__(self, env, action_noise_std=0.0):
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self.phase = "move_to_pre_block"
        self._action_noise_std = action_noise_std

    def get_theta_from_vector(self, vector):
        return np.arctan2(vector[1], vector[0])

    def theta_to_rotation2d(self, theta):
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return r

    def rotate(self, theta, xy_dir_block_to_ee):
        rot_2d = self.theta_to_rotation2d(theta)
        return rot_2d @ xy_dir_block_to_ee

    def _get_action_info(self, data, block, target):
        xy_block = data["%s_translation" % block][:2]
        theta_block = data["%s_orientation" % block]
        xy_target = data["%s_translation" % target][:2]
        xy_ee = data["effector_target_translation"][:2]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(xy_block_to_target)
        theta_to_target = self.get_theta_from_vector(xy_dir_block_to_target)

        theta_error = theta_to_target - theta_block
        # Block has 4-way symmetry.
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2.0
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2.0

        xy_pre_block = xy_block + -xy_dir_block_to_target * 0.05
        xy_nexttoblock = xy_block + -xy_dir_block_to_target * 0.03
        xy_touchingblock = xy_block + -xy_dir_block_to_target * 0.01
        xy_delta_to_nexttoblock = xy_nexttoblock - xy_ee
        xy_delta_to_touchingblock = xy_touchingblock - xy_ee

        xy_block_to_ee = xy_ee - xy_block
        xy_dir_block_to_ee = xy_block_to_ee / np.linalg.norm(xy_block_to_ee)

        theta_threshold_to_orient = 0.2
        theta_threshold_flat_enough = 0.03
        return pushing_info_module.PushingInfo(
            xy_block=xy_block,
            xy_ee=xy_ee,
            xy_pre_block=xy_pre_block,
            xy_delta_to_nexttoblock=xy_delta_to_nexttoblock,
            xy_delta_to_touchingblock=xy_delta_to_touchingblock,
            xy_dir_block_to_ee=xy_dir_block_to_ee,
            theta_threshold_to_orient=theta_threshold_to_orient,
            theta_threshold_flat_enough=theta_threshold_flat_enough,
            theta_error=theta_error,
        )

    def _get_move_to_block(self, xy_delta_to_nexttoblock, theta_threshold_to_orient, theta_error):
        diff = np.linalg.norm(xy_delta_to_nexttoblock)
        if diff < 0.001:
            self.phase = "push_block"
        # If need to re-oorient, then re-orient.
        if theta_error > theta_threshold_to_orient:
            self.phase = "orient_block_left"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "orient_block_right"
        # Otherwise, push into the block.
        xy_delta = xy_delta_to_nexttoblock
        return xy_delta

    def _get_push_block(self, theta_error, theta_threshold_to_orient, xy_delta_to_touchingblock):
        # If need to reorient, go back to move_to_pre_block, move_to_block first.
        if theta_error > theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        xy_delta = xy_delta_to_touchingblock
        return xy_delta

    def _get_orient_block_left(
        self,
        xy_dir_block_to_ee,
        orient_circle_diameter,
        xy_block,
        xy_ee,
        theta_error,
        theta_threshold_flat_enough,
    ):
        xy_dir_block_to_ee = self.rotate(0.2, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee
        if theta_error < theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta

    def _get_orient_block_right(
        self,
        xy_dir_block_to_ee,
        orient_circle_diameter,
        xy_block,
        xy_ee,
        theta_error,
        theta_threshold_flat_enough,
    ):
        xy_dir_block_to_ee = self.rotate(-0.2, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee
        if theta_error > -theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta


class MultimodalOrientedPushOracle(OrientedPushOracle):
    """Oracle for multimodal pushing task."""

    def __init__(self, env, goal_dist_tolerance=0.04, action_noise_std=0.0):
        super(MultimodalOrientedPushOracle, self).__init__(env)
        self._goal_dist_tolerance = goal_dist_tolerance
        self._action_noise_std = action_noise_std
        self._is_first = True

    def reset(self):
        self.origin = None
        self.first_preblock = None
        self._is_first = True
        self.phase = "move_to_pre_block"
        print("Resetting oracle.")

    def _get_move_to_preblock(self, xy_pre_block, xy_ee):
        max_step_velocity = 0.3
        # Go 5 cm away from the block, on the line between the block and target.
        xy_delta_to_preblock = xy_pre_block - xy_ee
        diff = np.linalg.norm(xy_delta_to_preblock)
        if diff < 0.001:
            self.phase = "move_to_block"
            if self.first_preblock is None:
                self.first_preblock = np.copy(xy_pre_block)
        xy_delta = xy_delta_to_preblock
        return xy_delta, max_step_velocity

    def _get_action_for_block_target(self, data, block="block", target="target"):
        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.35

        info = self._get_action_info(data, block, target)

        if self.origin is None:
            self.origin = np.copy(info.xy_ee)

        if self.phase == "move_to_pre_block":
            xy_delta, max_step_velocity = self._get_move_to_preblock(info.xy_pre_block, info.xy_ee)

        if self.phase == "return_to_first_preblock":
            max_step_velocity = 0.3
            if self.first_preblock is None:
                self.first_preblock = self.origin
            # Return to the first preblock.
            xy_delta_to_origin = self.first_preblock - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "return_to_origin"
            xy_delta = xy_delta_to_origin

        if self.phase == "return_to_origin":
            max_step_velocity = 0.3
            # Go 5 cm away from the block, on the line between the block and target.
            xy_delta_to_origin = self.origin - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "move_to_pre_block"
            xy_delta = xy_delta_to_origin

        if self.phase == "move_to_block":
            xy_delta = self._get_move_to_block(
                info.xy_delta_to_nexttoblock,
                info.theta_threshold_to_orient,
                info.theta_error,
            )

        if self.phase == "push_block":
            xy_delta = self._get_push_block(
                info.theta_error,
                info.theta_threshold_to_orient,
                info.xy_delta_to_touchingblock,
            )

        orient_circle_diameter = 0.025

        if self.phase == "orient_block_left" or self.phase == "orient_block_right":
            max_step_velocity = 0.15

        if self.phase == "orient_block_left":
            xy_delta = self._get_orient_block_left(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self.phase == "orient_block_right":
            xy_delta = self._get_orient_block_right(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self._action_noise_std != 0.0:
            xy_delta += self._np_random_state.randn(2) * self._action_noise_std

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance
        return xy_delta

    def _choose_goal_order(self):
        """Chooses block->target order for multimodal pushing."""
        # Define all possible ((first_block, first_target),
        # (second_block, second_target)).
        # 初期化時のIDが固定ならば，blockとtargetの組み合わせは固定するべき．タスク設定がそうじゃない．マルチモーダルってそういうこと．
        possible_orders = [
            (("block", "target"), ("block2", "target2")),
            (("block", "target2"), ("block2", "target")),
            (("block2", "target"), ("block", "target2")),
            (("block2", "target2"), ("block", "target")),
        ]
        # import pdb; pdb.set_trace()
        result = random.choice(possible_orders)
        # result = possible_orders[self._env._rng.choice(len(possible_orders))]
        return result

    def _action(self, _data):
        if self._is_first:
            (
                (self._first_block, self._first_target),
                (self._second_block, self._second_target),
            ) = self._choose_goal_order()
            self._current_block, self._current_target = (
                self._first_block,
                self._first_target,
            )
            self._is_first = False
            self._has_switched = False

        def _block_target_dist(block, target):
            dist = np.linalg.norm(_data["%s_translation" % block] - _data["%s_translation" % target])
            return dist

        if (
            _block_target_dist(self._first_block, self._first_target) < self._goal_dist_tolerance
            and not self._has_switched
        ):
            # If first block has been pushed to first target, switch to second block.
            self._current_block, self._current_target = (
                self._second_block,
                self._second_target,
            )
            self._has_switched = True
            self.phase = "return_to_first_preblock"
        # print(self.phase, self._current_block, self._current_target)

        xy_delta = self._get_action_for_block_target(
            data=_data, block=self._current_block, target=self._current_target
        )

        return xy_delta


@click.command()
@click.option("-o", "--output", required=False)
@click.option("-hz", "--control_hz", default=10, type=int)
@click.option("-epi", "--episodes", default=1000, type=int)
@click.option("-ms", "--max_steps", default=150, type=int)
@click.option("-c", "--chunk_length", default=-1)
def main(output, control_hz, episodes, max_steps, chunk_length):
    env: gym.Env = gym.make("BlockPushMultimodal-v0", control_frequency=control_hz)  # default valueと一緒だけど......
    oracle = MultimodalOrientedPushOracle(env)
    replay_buffer = ReplayBuffer.create_empty_zarr()
    num_retry = 0

    while replay_buffer.n_episodes < episodes:
        obs_history = list()
        action_history = list()
        seed = replay_buffer.n_episodes + num_retry
        env.seed(seed)
        oracle.reset()
        obs = env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            if step > max_steps:
                num_retry += 1
                print(
                    f"Max steps reached {(max_steps)}. Retrying with a new seed. ({replay_buffer.n_episodes + num_retry})"
                )
                break

            # demo_pusht.pyの作法に修正．記録する最初のアクションはreset()で取得したobsを利用する．そして最後にstep()を実行する．
            obs_conca = np.concatenate(list(obs.values()), axis=-1)
            obs_history.append(obs_conca)
            action = oracle._action(obs)
            action_history.append(action)

            obs, reward, done, info = env.step(action)
            # env.render()

        obs_history = np.array(obs_history)
        action_history = np.array(action_history)
        episode = {"obs": obs_history, "action": action_history}
        replay_buffer.add_episode(episode)

    replay_buffer.save_to_path(output, chunk_length=chunk_length)


if __name__ == "__main__":
    main()
