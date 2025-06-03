import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass

from mani_skill.sensors.camera import parse_camera_configs

from PIL import Image
import torch

from scipy.spatial.transform import Rotation as R

from mani_skill.utils.building import actors

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ArrayCan-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """The robot to use. Robot setups supported for teleop in this script are panda and panda_stick"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

def parse_args() -> Args:
    return tyro.cli(Args)

def pixel_to_camera_coords(u, v, Z, K):
    """
    Convert pixel coordinates to camera coordinates
    :param u: x pixel coordinate
    :param v: y pixel coordinate
    :param Z: depth value
    :param K: intrinsic matrix
    :return: camera coordinates (x, y, z)
    """
    # Ensure Z is a CPU tensor
    if torch.is_tensor(Z):
        Z = Z.cpu().item()
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def camera_to_world(p_cam, extrinsic):
    """
    Convert camera coordinates to world coordinates.
    
    Parameters:
        p_cam (np.ndarray): 3D point in camera coordinates (shape: [3])
        extrinsic (np.ndarray): 3x4 extrinsic matrix
        
    Returns:
        np.ndarray: 3D point in world coordinates (shape: [3])
    """

    p_cam = np.array(p_cam).reshape(3, 1)

    # Convert extrinsic to NumPy if it's a tensor
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()

    R = extrinsic[:, :3]
    t = extrinsic[:, 3:]

    # Inverse transform: p_world = R^T @ (p_cam - t)
    p_world = R.T @ (p_cam - t)
    return p_world.flatten()

# Set the camera pose
scipy_quat = R.from_euler("xyz", [0.0, np.pi / 2, 0]).as_quat()
quat_cam = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

_marker_counter = 0

def mark_can(scene, pose, radius=0.01):
    '''
    Marks the center of the top of the coke can with a blue sphere
    '''
    global _marker_counter
    name = f"marker_{_marker_counter}"
    _marker_counter += 1

    pose = sapien.Pose(p=[pose[0], pose[1], pose[2]])

    can_marker = actors.build_sphere(
        scene=scene,
        radius=radius,
        color=[0, 0, 1, 1],
        name=name,
        body_type="kinematic",
        add_collision=False,
        initial_pose=pose
    )
    return can_marker

def mark_dst(scene, pose, radius=0.01):
    '''
    Marks the destination of the coke can with a green sphere
    '''
    global _marker_counter
    name = f"marker_{_marker_counter}"
    _marker_counter += 1

    pose = sapien.Pose(p=[pose[0], pose[1], pose[2]])

    dst_marker = actors.build_sphere(
        scene=scene,
        radius=radius,
        color=[0, 1, 0, 1],
        name=name,
        body_type="kinematic",
        add_collision=False,
        initial_pose=pose
    )
    return dst_marker


def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
        sensor_configs={
            "base_camera": dict(
                width=512,
                height=512,
                pose=sapien.Pose(p=[0.0, 0.0, 0.6], q=quat_cam),
            )
        }
    )
    
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)

    # Capture a screenshot of the scene
    # env.unwrapped.scene.update_render()
    # camera = env.unwrapped._sensors["base_camera"]
    # camera.capture()
    # obs_dict = camera.get_obs(rgb=True, depth=True)
    # images = camera.get_images(obs_dict)

    # Save the screenshot as a RGB image
    # rgb_image = images["rgb"]
    # rgb_np = rgb_image.squeeze().cpu().numpy()
    # img = Image.fromarray(rgb_np)
    # path = "/home/jypaulsung/Sapien/only_coke_1.png"
    # img.save(path)

    pixels = [(236, 300), (255, 172), (354, 338)] # a point on the circle that represents the top of the coke can (at 0 degrees)

    # Get depth map
    depth_map = obs_dict["depth"].squeeze()

    # Get camera intrinsic matrix
    params = camera.get_params()
    K = params["intrinsic_cv"].squeeze()
    extrinsic = params["extrinsic_cv"].squeeze()
 
    # Get camera pose
    camera_pose = camera.config.pose
 
    # Convert each pixel to world coordinates
    world_coords = []
    for (u, v) in pixels:
        u_int, v_int = int(round(u)), int(round(v))
        Z = depth_map[v_int, u_int].cpu().item()
        Z = Z / 1000.0 # convert to meters
        p_cam = pixel_to_camera_coords(u, v, Z, K)
        p_world = camera_to_world(p_cam, extrinsic)
        p_world[1] += 0.022 # add a small offset to the y-coordinate so that p_world points to the center of the top of the coke can
        world_coords.append(p_world)

    # Print the world coordinates
    for i, coord in enumerate(world_coords):
        print(f"Coke can {i+1} world coordinate: [{coord[0]}, {coord[1]}, {coord[2]}]")
    
    # Visualize the world coordinates
    for coord in world_coords:
        # Ensure coord is a 1D NumPy array with 3 elements
        if isinstance(coord, torch.Tensor):
            position = coord.squeeze().cpu().numpy()[:3].astype(np.float32)
        else:
            position = np.array(coord[:3], dtype=np.float32)
        
        # Create the Pose with the corrected position
        pose = sapien.Pose(p=position)
        
        # Add the visual sphere markers
        mark_can(env.unwrapped.scene, pose=pose.p)

    # Add the destination marker
    dst_coords = [[-0.03275781503412573, 0.11699218302965164, 0.105], [-0.03275781503412573, 0.0, 0.105], [-0.03275781503412573, -0.11699218302965164, 0.105]]
    for i, coord in enumerate(dst_coords):
        # Ensure coord is a 1D NumPy array with 3 elements
        if isinstance(coord, torch.Tensor):
            position = coord.squeeze().cpu().numpy()[:3].astype(np.float32)
        else:
            position = np.array(coord[:3], dtype=np.float32)
        
        # Create the Pose with the corrected position
        pose = sapien.Pose(p=position)
        
        # Add the visual sphere markers
        mark_dst(env.unwrapped.scene, pose=pose.p)

        # Print the destination coordinates
        print(f"Destination coordinate of coke can {i+1}: {coord}")

    env.unwrapped.scene.update_render()

    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path

    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env

def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = False
    if env.unwrapped.robot_uids == "panda_stick":
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    elif env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
        robot_has_gripper = True
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_panda_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    select_panda_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the panda hand up
            j: move the panda hand down
            arrow_keys: move the panda hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            if env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)
            elif env.unwrapped.robot_uids == "panda_stick":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.15]), dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False



    return args
if __name__ == "__main__":
    main(parse_args())
