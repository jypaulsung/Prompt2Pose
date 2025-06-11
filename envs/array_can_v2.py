# add to python3.11/site-packages/mani_skill/envs/tasks/tabletop/array_can_v2.py
'''
This script implements an ArrayCan-v2 environment for the starting_coordinates.py.
'''
from typing import Any, Dict, Union

import numpy as np
import random
import sapien
import torch
import os

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose



@register_env("ArrayCan-v2", max_episode_steps=50)
class ArrayCanEnv2(BaseEnv):
    """
    **Task Description:**
    The goal is to arrange cans in a specific pattern on the table.
    The cans should be equally spaced and form a linear arrangement.
    
    **Randomizations:**
    - The cans are placed randomly on the table within a specified region.
    - The cans have their xy positions on top of the table scene randomized. The positions are sampled such that the cans do not collide with each other.
    
    **Success Conditions:**
    - The cans are arranged in a linear pattern with equal spacing.
    - The cans are not being grasped by the robot (robot must let go of the cans).
    - The cans are static and not moving.
    """
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_cans = random.randint(3, 5)  # Randomly choose number of cans between 3 and 5
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.7, 0, 0.3], target=[0.18, 0, 0])
        return [CameraConfig("base_camera", pose, 512, 512, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        self.scale = 0.03  # modeling scale for the cans
        self.radius = 1  # can radius (m)
        self.height = 3.5   # can total height (m)

        self.scaled_radius = self.radius * self.scale
        self.scaled_half_height = (self.height * self.scale) / 2

        mass = 0.5
        I_z = 0.5 * mass * self.scaled_radius**2
        I_xy = mass * (2*self.scaled_half_height)**2 / 3

        builder = self.scene.create_actor_builder()
        model_path = os.path.join(os.path.dirname(__file__), "/home/jypaulsung/Sapien/Shared/models/coke/coke.obj")
        builder.add_visual_from_file(
            model_path,
            scale=[self.scale] * 3
        )
        builder.add_convex_collision_from_file(model_path, scale=[self.scale]*3)
        builder.set_mass_and_inertia(mass, sapien.Pose([0,0,self.scaled_half_height*(-0.1)],[1,0,0,0]), np.array([I_xy,I_xy,I_z], dtype=np.float32))

        self.cokes: List[sapien.Actor] = []
        for i in range(self.num_cans):
            coke = builder.build(name=f"coke{i}")
            self.cokes.append(coke)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.2, -0.2], [0.12, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            for can in self.cokes:
                xy_sample = xy + sampler.sample(self.scaled_radius, 100, verbose=False)
                xyz[:, :2] = xy_sample
                
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                can.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))


    def evaluate(self):

        pos_list = [can.pose.p for can in self.cokes]
        coke_pos = torch.stack(pos_list, dim=1)
        dists = torch.cdist(coke_pos, coke_pos)
        triu_inds = torch.triu_indices(self.num_cans, self.num_cans, offset=1)
        pairwise = dists[:, triu_inds[0], triu_inds[1]]
        dis, _ = torch.sort(pairwise, dim=-1)
        d0, d1, d2 = dis.unbind(-1)[:3]
        tol = 0.01
        is_equally_spaced = ((torch.abs(d0 - d1) <= tol) | (torch.abs(d1 - d2) <= tol))

        if self.num_cans == 3:
            pos_A, pos_B, pos_C = pos_list
            area = 0.5 * (
                pos_A[:, 0] * (pos_B[:, 1] - pos_C[:, 1]) +
                pos_B[:, 0] * (pos_C[:, 1] - pos_A[:, 1]) +
                pos_C[:, 0] * (pos_A[:, 1] - pos_B[:, 1])
            )
            lin_tol = 0.001
            is_linear = area.abs() <= lin_tol
        else:
            is_linear = torch.ones_like(is_equally_spaced)

        grasp_flags = torch.stack([self.agent.is_grasping(can) for can in self.cokes], dim=1)

        is_can_grasped = grasp_flags.any(dim=1)
        success = is_equally_spaced & is_linear & (~is_can_grasped)

        return {
            "is_can_grasped": is_can_grasped,
            "is_equally_spaced": is_equally_spaced,
            "is_linear": is_linear,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            for idx, can in enumerate(self.cokes):
                obs[f"coke{idx}_pose"] = can.pose.raw_pose
                obs[f"tcp_to_coke{idx}_pos"] = can.pose.p - self.agent.tcp.pose.p
            for i in range(self.num_cans):
                for j in range(i + 1, self.num_cans):
                    obs[f"coke{i}_to_coke{j}_pos"] = (
                        self.cokes[j].pose.p - self.cokes[i].pose.p
                    )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        coke_pos = torch.stack([can.pose.p for can in self.cokes], dim=1)
        coke_to_tcp_dists = torch.linalg.norm(tcp_pose.unsqueeze(1) - coke_pos, dim=-1)
        min_dist, _ = torch.min(coke_to_tcp_dists, dim=1)
        reward = 2 * (1 - torch.tanh(5 * min_dist))


        grasped = info["is_can_grasped"].float()
        spaced  = info["is_equally_spaced"].float()
        lined   = info["is_linear"].float()
        success = info["success"].float()

        w_grasp  = 1.0
        w_space  = 2.0
        w_line   = 3.0
        w_succ   = 6.0

        reward = reward + grasped * w_grasp + spaced  * w_space \
                        + lined * w_line + success * w_succ

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
    
    def get_can_poses(self):
        return [can.pose.p for can in self.cokes]

