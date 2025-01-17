from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class UnetCompositionPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            model1: ConditionalUnet1D,
            model2: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            normalizer1: LinearNormalizer,
            normalizer2: LinearNormalizer,
            obs_encoder1: MultiImageObsEncoder,
            obs_encoder2: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder1.output_shape()[0]

        # create diffusion model この要素は学習時に設定されているのでは？つまりいらない．
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # model = ConditionalUnet1D(
        #     input_dim=input_dim,
        #     local_cond_dim=None,
        #     global_cond_dim=global_cond_dim,
        #     diffusion_step_embed_dim=diffusion_step_embed_dim,
        #     down_dims=down_dims,
        #     kernel_size=kernel_size,
        #     n_groups=n_groups,
        #     cond_predict_scale=cond_predict_scale
        # )

        self.obs_encoder1 = obs_encoder1
        self.obs_encoder2 = obs_encoder2
        self.model1 = model1
        self.model2 = model2
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        # self.normalizer = LinearNormalizer()
        self.normalizer1 = normalizer1
        self.normalizer2 = normalizer2
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond1=None, global_cond1=None,
            local_cond2=None, global_cond2=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model1 = self.model1
        model2 = self.model2
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output1 = model1(trajectory, t,
                local_cond=local_cond1, global_cond=global_cond1)

            model_output2 = model2(trajectory, t,
                local_cond=local_cond2, global_cond=global_cond2)

            model_output = (model_output1 + model_output2) / 2 # 合成
            # model_output = model_output1*0.75 + model_output2*0.25
            # model_output = model_output1*0.25 + model_output2*0.75
            # model_output = model_output1*0.10 + model_output2*0.90
            # model_output = model_output1

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs1 = self.normalizer1.normalize(obs_dict)
        nobs2 = self.normalizer2.normalize(obs_dict)
        value1 = next(iter(nobs1.values()))
        value2 = next(iter(nobs2.values()))
        B, To = value1.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond1, local_cond2 = None, None
        global_cond1, global_cond2 = None, None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs1 = dict_apply(nobs1, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            this_nobs2 = dict_apply(nobs2, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features1 = self.obs_encoder1(this_nobs1)
            nobs_features2 = self.obs_encoder2(this_nobs2)
            # reshape back to B, Do
            global_cond1 = nobs_features1.reshape(B, -1)
            global_cond2 = nobs_features2.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            assert False, "not used this time"
            # condition through impainting
            this_nobs1 = dict_apply(nobs1, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            this_nobs2 = dict_apply(nobs2, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features1 = self.obs_encoder(this_nobs1)
            nobs_features2 = self.obs_encoder(this_nobs2)
            # reshape back to B, T, Do
            nobs_features1 = nobs_features1.reshape(B, To, -1)
            nobs_features2 = nobs_features2.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features1
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond1=local_cond1,
            global_cond1=global_cond1,
            local_cond2=local_cond2,
            global_cond2=global_cond2,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer1['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
