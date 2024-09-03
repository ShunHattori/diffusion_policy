import collections
import math
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv
from gym.wrappers import FlattenObservation

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.block_pushing_mod.block_pushing_multimodal import (
    BlockPushMultimodal,
)
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.real_world.rtde_interpolation_controller import (
    RTDEInterpolationController,
)


class BlockPushModLowdimRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=5,
        crf=22,
        past_action=False,
        abs_action=False,
        obs_eef_target=True,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)

        self.task_fps = 3
        fps = 3
        steps_per_render = 1

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    FlattenObservation(
                        BlockPushMultimodal(
                            control_frequency=self.task_fps, shared_memory=False, seed=seed, abs_action=abs_action
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf, thread_type="FRAME", thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        # UR robot setup
        shm_manager = SharedMemoryManager()
        shm_manager.start()
        max_pos_speed = 0.25
        max_rot_speed = 0.6
        tcp_offset = 0.13
        max_obs_buffer_size = 30
        cube_diag = np.linalg.norm([1, 1, 1])
        j_init = np.array([0, -90.20, -118.42, -61.38, 90, -4.77]) / 180 * np.pi

        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip="172.17.0.2",
            frequency=125,  # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed * cube_diag,
            max_rot_speed=max_rot_speed * cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=True,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
        )
        self.robot = robot

        def init_fn(env,seed,enable_render):
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = pathlib.Path(output_dir).joinpath("media", wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env.env.file_path = filename

            # set seed
            assert isinstance(env, MultiStepWrapper)
            env.seed(seed)


        # Single environment for training
        seed = train_start_seed
        enable_render = True
        self.train_env = env_fn()
        init_fn(self.train_env, seed=seed, enable_render=enable_render)

        # Single environment for testing
        seed = test_start_seed
        enable_render = True
        self.test_env = env_fn()
        init_fn(self.test_env, seed=seed, enable_render=enable_render)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_eef_target = obs_eef_target

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        # Use train_env or test_env based on the requirement
        env = self.test_env

        # initialize
        obs = env.reset()
        policy.reset()
        self.robot.start(wait=True)
        self.robot.start_wait()  # 起動待ちで初期姿勢がget_stateに反映される

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc="Eval BlockPushModLowdimRunner",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )
        done = False
        perv_target_pose = self.robot.get_state()["TargetTCPPose"]

        step_duration = 1 / self.task_fps
        while not done:
            step_stime = time.time()

            # prepare observations
            if not self.obs_eef_target:
                obs[..., 8:10] = 0
            # print(f"obs: {obs}")
            np_obs_dict = {"obs": obs.astype(np.float32)}
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
            obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0))

            # predict actions
            with torch.no_grad():
                stime = time.time()
                action_dict = policy.predict_action(obs_dict)
            print(f"Prediction time: {time.time() - stime}")
            np_action_dict = dict_apply(action_dict, lambda x: x.cpu().numpy())
            action = np_action_dict["action"]

            # step environment
            action = action.squeeze(0)
            # print(f"action: {action}")
            # print(f"self.robot.get_state(): {self.robot.get_state()}")

            assert len(action) == 1
            this_target_pose = perv_target_pose.copy()
            this_target_pose[[0, 1]] += action[-1]
            perv_target_pose = this_target_pose
            this_target_poses = np.expand_dims(this_target_pose, axis=0)
            this_target_poses[:, :2] = np.clip(this_target_poses[:, :2], [0.25, -0.45], [0.77, 0.40])

            obs, reward, done, info = env.step(action)
            done = np.all(done)

            # set waypoint for robot
            this_target_poses = this_target_poses.squeeze(0)
            # print(f"this_target_poses: {this_target_poses}")
            self.robot.schedule_waypoint(pose=this_target_poses, target_time=time.time() + step_duration)
            # self.robot.servoL(pose=this_target_poses)

            elapsed_time = time.time() - step_stime
            sleep_duration = step_duration - elapsed_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            pbar.update(1)
        pbar.close()

        # stop robot
        self.robot.stop(wait=True)

        # gather results and log
        video_path = env.render()
        total_reward = 1 #np.sum(env.call("get_attr", "reward"))
        log_data = {
            "video": wandb.Video(video_path) if video_path else None,
            "total_reward": total_reward,
        }

        return log_data
