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


@register_env("ArrayCan-v1", max_episode_steps=50)
class ArrayCanEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

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
        
        self.scale = 0.03  # 모델 스케일
        self.radius = 1  # 캔 반지름 (m)
        self.height = 3.5   # 캔 전체 높이 (m)


        self.scaled_radius = self.radius * self.scale
        self.scaled_half_height = (self.height * self.scale) / 2

        mass = 0.5
        I_z = 0.5 * mass * self.scaled_radius**2
        I_xy = mass * (2*self.scaled_half_height)**2 / 3

        builder = self.scene.create_actor_builder()
        model_path = os.path.join(os.path.dirname(__file__), "../../../assets/models/coke/coke.obj")
        builder.add_visual_from_file(
            model_path,
            scale=[self.scale] * 3
        )
        builder.add_convex_collision_from_file(model_path, scale=[self.scale]*3)
        builder.set_mass_and_inertia(mass, sapien.Pose([0,0,self.scaled_half_height*(-0.1)],[1,0,0,0]), np.array([I_xy,I_xy,I_z], dtype=np.float32))

        self.cokeA = builder.build(name="cokeA")
        self.cokeB = builder.build(name="cokeB")
        self.cokeC = builder.build(name="cokeC")


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.15, -0.15], [0.15, 0.15]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            for can in (self.cokeA, self.cokeB, self.cokeC):
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
        pos_A = self.cokeA.pose.p
        pos_B = self.cokeB.pose.p
        pos_C = self.cokeC.pose.p

        d_AB = torch.linalg.norm(pos_A - pos_B, axis=1)
        d_BC = torch.linalg.norm(pos_B - pos_C, axis=1)
        d_CA = torch.linalg.norm(pos_C - pos_A, axis=1)

        dis = torch.stack([d_AB, d_BC, d_CA], dim=-1)
        dis, _ = torch.sort(dis, dim=-1)
        d0,d1,d2 = dis.unbind(-1)

        tol = 0.01
        is_equally_spaced = ((torch.abs(d0-d1) <= tol) |
                             (torch.abs(d1-d2) <= tol))
        
        eps = 1e-6
        lin_tol = 0.001
        # lin_AB = (pos_B[:,1] - pos_A[:,1]) / (pos_B[:,0] - pos_A[:,0] + eps)
        # lin_AC = (pos_C[:,1] - pos_A[:,1]) / (pos_C[:,0] - pos_A[:,0] + eps)
        area = 0.5*(pos_A[:,0]*(pos_B[:,1]-pos_C[:,1])+pos_B[:,0]*(pos_C[:,1]-pos_A[:,1])+pos_C[:,0]*(pos_A[:,1]-pos_B[:,1]))
        # is_linear = torch.abs(lin_AB - lin_AC) <= lin_tol
        is_linear = area <= lin_tol

        is_can_grasped = (self.agent.is_grasping(self.cokeA) |
                          self.agent.is_grasping(self.cokeB) |
                          self.agent.is_grasping(self.cokeC))
        
        success = is_equally_spaced * is_linear * (~is_can_grasped)

        return {
            "is_can_grasped": is_can_grasped,
            "is_equally_spaced": is_equally_spaced,
            "is_linear": is_linear,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cokeA_pose=self.cokeA.pose.raw_pose,
                cokeB_pose=self.cokeB.pose.raw_pose,
                cokeC_pose=self.cokeC.pose.raw_pose,
                tcp_to_cokeA_pos=self.cokeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cokeB_pos=self.cokeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cokeC_pos=self.cokeC.pose.p - self.agent.tcp.pose.p,
                cokeA_to_cokeB_pos=self.cokeB.pose.p - self.cokeA.pose.p,
                cokeB_to_cokeC_pos=self.cokeC.pose.p - self.cokeB.pose.p,
                cokeC_to_cokeA_pos=self.cokeA.pose.p - self.cokeC.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        coke_pos = torch.stack([
            self.cokeA.pose.p,
            self.cokeB.pose.p,
            self.cokeC.pose.p
        ], dim=1)
        coke_to_tcp_dists = torch.linalg.norm(tcp_pose.unsqueeze(1) - coke_pos, dim=-1)
        min_dist, _ = torch.min(coke_to_tcp_dists, dim=1)
        reward = 2 * (1 - torch.tanh(5 * min_dist))

        grasped = info["is_can_grasped"].float()       # 캔을 잡고 있는지
        spaced  = info["is_equally_spaced"].float()    # 동일 간격 조건
        lined   = info["is_linear"].float()            # 일직선 조건
        success = info["success"].float()              # 최종 성공

        w_grasp  = 1.0   # grasp 단계 보너스
        w_space  = 2.0   # spacing 단계 보너스
        w_line   = 3.0   # alignment 단계 보너스
        w_succ   = 6.0   # full success 보너스

        reward = reward + grasped * w_grasp + spaced  * w_space \
                        + lined * w_line + success * w_succ


        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
