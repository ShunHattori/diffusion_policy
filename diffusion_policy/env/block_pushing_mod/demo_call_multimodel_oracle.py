import gym

"""Oracle for multimodal pushing task."""
import random
from collections import OrderedDict

import click
import numpy as np

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import
import pygame

import diffusion_policy.env.block_pushing_mod.oracles.pushing_info as pushing_info_module
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.block_pushing_mod import block_pushing_multimodal
from diffusion_policy.env.block_pushing_mod.utils.utils_pybullet import (
    ObjState,
    XarmState,
)
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
        # 初期化時のIDが固定ならば，blockとtargetの組み合わせは固定するべき．
        possible_orders = [
            # (("block", "target"), ("block2", "target2")),
            (("block", "target2"), ("block2", "target")),
            (("block2", "target"), ("block", "target2")),
            # (("block2", "target2"), ("block", "target")),
        ]
        # import pdb; pdb.set_trace()
        result = random.choice(possible_orders)
        # result = possible_orders[self._env._rng.choice(len(possible_orders))]
        return result

    def _action(self, _data):
        if self._is_first:
            self.reset()
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

        # return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))
        return xy_delta


@click.command()
@click.option("-o", "--output", required=False)
@click.option("-hz", "--control_hz", default=10, type=int)
def environment_test(output, control_hz):
    env = gym.make("BlockPushMultimodal-v0", control_frequency=control_hz)  # default valueと一緒だけど......
    env.reset()
    oracle = MultimodalOrientedPushOracle(env)

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    # キーアクセスのテスト
    # 　関節角度はjointstateに入っている！
    # getJointStateのJointPositionが関節角度！
    # IKの解と一致していることを確認ずみ (https://dirkmittler.homeip.net/blend4web_ce/uranium/bullet/docs/pybullet_quickstartguide.pdf)

    # シミュレーターの周期はcontrol_hzとstepで管理されているから，
    # Oracleには源氏の時刻の状態を渡す．
    #  内部では辞書形式で値を取得しているので，それに従う．
    # Key のリストアップ %sにはpossible_ordersの文字列が入る（block,target,block2,target2）
    # block_translation, block_orientation, target_translation, target_orientation, block2_translation, block2_orientation, target2_translation, target2_orientation, effector_target_translation
    # %s_translationは3次元ベクトル，%s_orientationはyaw軸の角度θ, effector_target_translationはEEを位置を意味する3次元ベクトル
    #
    for i in range(350):
        # while True:
        obs_raw: dict = env.get_pybullet_state()
        _robot_data = XarmState.serialize(obs_raw["robots"][0])
        angles = tuple(_robot_data.get("joint_state")[i][0] for i in range(1, 7))  # 各関節の角度を取り出すコード
        objects = [ObjState.serialize(obs_raw["objects"][i]) for i in range(2)]
        targets = [ObjState.serialize(obs_raw["targets"][i]) for i in range(2)]
        effector = ObjState.serialize(obs_raw["robot_end_effectors"][0])

        def _get_theta_from_quat(quat):
            """
            getBasePositionAndOrientation returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
            Use getEulerFromQuaternion to convert the quaternion to Euler if needed.
            np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
            """
            return env.pybullet_client.getEulerFromQuaternion(quat)[2]

        data = {
            "block_translation": np.array(objects[0]["base_pose"][0]),
            "target_translation": np.array(targets[0]["base_pose"][0]),
            "block2_translation": np.array(objects[1]["base_pose"][0]),
            "target2_translation": np.array(targets[1]["base_pose"][0]),
            "block_orientation": np.array(_get_theta_from_quat(objects[0]["base_pose"][1])),
            "target_orientation": np.array(_get_theta_from_quat(targets[0]["base_pose"][1])),
            "block2_orientation": np.array(_get_theta_from_quat(objects[1]["base_pose"][1])),
            "target2_orientation": np.array(_get_theta_from_quat(targets[1]["base_pose"][1])),
            "effector_target_translation": np.array(effector["base_pose"][0]),
        }

        action = oracle._action(data)
        next_state, reward, done, info = env.step(action)

        log = {
            "angles": angles,
            "robot_end_effectors": [ObjState.serialize(obs_raw["robot_end_effectors"][0])],
            "targets": [ObjState.serialize(obs_raw["targets"][i]) for i in range(2)],
            "objects": [ObjState.serialize(obs_raw["objects"][i]) for i in range(2)],
        }

        env.render()

        if done:
            break
    env.close()


@click.command()
@click.option("-o", "--output", required=False)
@click.option("-hz", "--control_hz", default=10, type=int)
@click.option("-epi", "--episodes", default=350, type=int)
def main(output, control_hz, episodes):
    env: gym.Env = gym.make("BlockPushMultimodal-v0", control_frequency=control_hz)  # default valueと一緒だけど......
    oracle = MultimodalOrientedPushOracle(env)
    replay_buffer = ReplayBuffer.create_empty_zarr()

    for _ in range(episodes):
        episode = list()

        seed = replay_buffer.n_episodes
        env.seed(seed)
        obs = env.reset()
        done = False

        while not done:
            # obs_raw: dict = env.get_pybullet_state()
            # _robot_data = XarmState.serialize(obs_raw["robots"][0])
            # angles = tuple(_robot_data.get("joint_state")[i][0] for i in range(1, 7))  # 各関節の角度を取り出すコード
            # objects = [ObjState.serialize(obs_raw["objects"][i]) for i in range(2)]
            # targets = [ObjState.serialize(obs_raw["targets"][i]) for i in range(2)]
            # effector = ObjState.serialize(obs_raw["robot_end_effectors"][0])

            # def _get_theta_from_quat(quat):
            #     """
            #     getBasePositionAndOrientation returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
            #     Use getEulerFromQuaternion to convert the quaternion to Euler if needed.
            #     np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
            #     """
            #     return env.pybullet_client.getEulerFromQuaternion(quat)[2]

            # data = {
            #     "block_translation": np.array(objects[0]["base_pose"][0], dtype=np.float32)[:2],
            #     "target_translation": np.array(targets[0]["base_pose"][0], dtype=np.float32)[:2],
            #     "block2_translation": np.array(objects[1]["base_pose"][0], dtype=np.float32)[:2],
            #     "target2_translation": np.array(targets[1]["base_pose"][0], dtype=np.float32)[:2],
            #     "effector_target_translation": np.array(effector["base_pose"][0], dtype=np.float32)[:2],
            #     "block_orientation": np.array(
            #         _get_theta_from_quat(objects[0]["base_pose"][1]), dtype=np.float32, ndmin=1
            #     ),
            #     "target_orientation": np.array(
            #         _get_theta_from_quat(targets[0]["base_pose"][1]), dtype=np.float32, ndmin=1
            #     ),
            #     "block2_orientation": np.array(
            #         _get_theta_from_quat(objects[1]["base_pose"][1]), dtype=np.float32, ndmin=1
            #     ),
            #     "target2_orientation": np.array(
            #         _get_theta_from_quat(targets[1]["base_pose"][1]), dtype=np.float32, ndmin=1
            #     ),
            # }

            action = oracle._action(obs)
            obs, reward, done, info = env.step(action)

            log = {}
            log["obs"] = np.concatenate([np.array(value, dtype=np.float32) for value in obs.values()], axis=0)
            log["action"] = np.array(action, dtype=np.float32)
            episode.append(log)

            # env.render()

        data_dict = {}
        for key in episode[0].keys():
            data_dict[key] = np.stack([log[key] for log in episode])
        replay_buffer.add_episode(data_dict, compressors="disk")

    replay_buffer.save_to_path(output)


if __name__ == "__main__":
    # environment_test()
    main()


"""
robots: [XarmState(obj_id=2, base_pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), base_vel=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), joint_info=((0, 'world_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 'link_base', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1), (1, 'shoulder_pan_joint', 0, 7, 6, 1, 0.0, 0.0, -6.28318530718, 6.28318530718, 150.0, 3.15, 'shoulder_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.089159), (0.0, 0.0, 0.0, 1.0), 0), (2, 'shoulder_lift_joint', 0, 8, 7, 1, 0.0, 0.0, -6.28318530718, 6.28318530718, 150.0, 3.15, 'upper_arm_link', (0.0, 1.0, 0.0), (0.0, 0.13585, 0.0), (0.0, -0.7071067811848163, 0.0, 0.7071067811882787), 1), (3, 'elbow_joint', 0, 9, 8, 1, 0.0, 0.0, -3.14159265359, 3.14159265359, 150.0, 3.15, 'forearm_link', (0.0, 1.0, 0.0), (0.0, -0.1197, 0.14499999999999996), (0.0, 0.0, 0.0, 1.0), 2), (4, 'wrist_1_joint', 0, 10, 9, 1, 0.0, 0.0, -6.28318530718, 6.28318530718, 28.0, 3.2, 'wrist_1_link', (0.0, 1.0, 0.0), (0.0, 0.0, 0.14225), (0.0, -0.7071067811848163, 0.0, 0.7071067811882787), 3), (5, 'wrist_2_joint', 0, 11, 10, 1, 0.0, 0.0, -6.28318530718, 6.28318530718, 28.0, 3.2, 'wrist_2_link', (0.0, 0.0, 1.0), (0.0, 0.093, 0.0), (0.0, 0.0, 0.0, 1.0), 4), (6, 'wrist_3_joint', 0, 12, 11, 1, 0.0, 0.0, -6.28318530718, 6.28318530718, 28.0, 3.2, 'wrist_3_link', (0.0, 1.0, 0.0), (0.0, 0.0, 0.09465), (0.0, 0.0, 0.0, 1.0), 5), (7, 'ee_fixed_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 'ee_link', (0.0, 0.0, 0.0), (0.0, 0.0823, 0.0), (0.706825181105366, 0.0, 0.0, 0.7073882691671998), 6)), joint_state=((0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), (-1.1446004126283698, -0.0003258301956876111, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.01089444856172156), (-1.1236436943420285, -8.738691868750981e-05, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -33.1464272078709), (2.127131245281684, 0.0001649040160241963, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -12.202767015090666), (-2.5734669380330564, -0.00019386882546472417, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -1.302901037030199), (-1.5705325187505115, -5.840769810394608e-05, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -0.047485795537207734), (0.42606691125311075, 0.00018333785422414844, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -0.00024232031773351818), (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)), target_effector_pose=Pose3d(rotation=<scipy.spatial.transform._rotation.Rotation object at 0x7f0b784ab3f0>, translation=array([ 0.3 , -0.4 ,  0.06])), goal_translation=None)]

robot_end_effectors: [ObjState(obj_id=3, base_pose=((0.30076074378525747, -0.40110500564859497, 0.05780186159253613), (0.0011386294688875745, 0.9999993082952141, -0.00028494789210420493, -7.574116856236417e-05)), base_vel=((-0.00046588220414291165, -0.0001517309511207036, 5.956232672098272e-06), (0.0006267072671231488, -0.00035382494112520195, 0.003133555185027067)), joint_info=((0, 'tipJoint', 4, -1, -1, 0, 10.0, 0.0, -6.28318530718, 6.28318530718, 150.0, 3.15, 'tipLink', (0.0, 0.0, 0.0), (0.0, 0.0, 0.029), (0.0, 0.0, 0.0, 1.0), -1),), joint_state=((0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),))]

targets: [ObjState(obj_id=4, base_pose=((0.515545715918793, 0.19738471220024115, 0.02), (0.0, 0.0, 0.9995577245240199, 0.029738112656381623)), base_vel=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), joint_info=(), joint_state=()), ObjState(obj_id=5, base_pose=((0.27642796222779115, 0.20231000392063286, 0.02), (0.0, 0.0, 0.9999891389449757, -0.0046606857957000244)), base_vel=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), joint_info=(), joint_state=())]

objects: [ObjState(obj_id=6, base_pose=((-0.250143214970878, -1.015025627604685, 0.01898593514213643), (2.0078257082841014e-06, 3.002891735276099e-05, -0.37234862792174966, 0.9280929362833353)), base_vel=((-0.5712375338666893, -0.18043267598915377, 0.00010244986164296283), (0.0005534838244221237, -0.0008169307859330992, 13.991970272800861)), joint_info=(), joint_state=()), ObjState(obj_id=7, base_pose=((0.38256554189917197, -0.25575052767455014, 0.01898509382544836), (3.948674424398224e-05, 4.188560606201641e-05, 0.8868206021577709, 0.4621138563978811)), base_vel=((1.3370629943951002e-05, 1.4828832783820478e-05, 9.325932797506903e-05), (-0.0006745653269397214, 0.0009891855223804298, -0.00017738288578947203)), joint_info=(), joint_state=())]

@classmethod
def calc_unnormalized_state(cls, norm_state):

    effector_target_translation = cls._unnormalize(
        norm_state["effector_target_translation"],
        EFFECTOR_TARGET_TRANSLATION_MIN,
        EFFECTOR_TARGET_TRANSLATION_MAX,
    )
    # Note: normalized state does not include effector_translation state, this
    # means this component will be missing (and is marked nan).
    effector_translation = np.array([np.nan, np.nan], np.float32)

    effector_target_to_block_translation = cls._unnormalize(
        norm_state["effector_target_to_block_translation"],
        EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN,
        EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX,
    )
    block_translation = effector_target_to_block_translation + effector_target_translation
    ori_cos_sin = cls._unnormalize(
        norm_state["block_orientation_cos_sin"],
        BLOCK_ORIENTATION_COS_SIN_MIN,
        BLOCK_ORIENTATION_COS_SIN_MAX,
    )
    block_orientation = np.array([math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32)

    effector_target_to_target_translation = cls._unnormalize(
        norm_state["effector_target_to_target_translation"],
        EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN,
        EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX,
    )
    target_translation = effector_target_to_target_translation + effector_target_translation
    ori_cos_sin = cls._unnormalize(
        norm_state["target_orientation_cos_sin"],
        TARGET_ORIENTATION_COS_SIN_MIN,
        TARGET_ORIENTATION_COS_SIN_MAX,
    )
    target_orientation = np.array([math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32)

    return collections.OrderedDict(
        block_translation=block_translation,
        block_orientation=block_orientation,
        effector_translation=effector_translation,
        effector_target_translation=effector_target_translation,
        target_translation=target_translation,
        target_orientation=target_orientation,
    )
"""
