"""
This script is used to collect the starting coordinates of coke cans in the ArrayCan-v2 environment.
The coordinates are saved in a JSON-like .txt file named `can_data_<seed>.txt` in the specified directory.
'can_data_<seed>.txt' contains the following information:
- circle_centers: List of detected circle centers in the image.
- circle_edges_0_deg: List of points on the detected circles at 0 degrees.
- world_coordinates: List of world coordinates of the detected cans. (center of top of the can)
"""

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
import random
import os
import time
import requests
import cv2
import math
import json
from transformers import GroundingDinoProcessor
from transformers import GroundingDinoForObjectDetection
import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R

from mani_skill.utils.building import actors

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ArrayCan-v2"
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

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# # colors for visualization
# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# def plot_results(pil_img, scores, labels, boxes, text, output_path):
#     """
#     Visualizes the results of the object detection and saves the image with bounding boxes and labels.
#     """
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         label = f'{text}: {score:0.2f}'
#         ax.text(xmin, ymin, label, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.savefig(output_path)

# def run_inference(image, text):
#     """
#     Runs inference on the image using the GroundingDino model and returns the outputs, processor (pre-trained GroundingDino), and device.
#     """
#     processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
#     inputs = processor(images=image, text=preprocess_caption(text), return_tensors="pt")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
#     model = model.to(device)

#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     if torch.cuda.is_available():
#         print("Current device:", torch.cuda.current_device())
#         print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print("Using CPU for inference.")

#     with torch.no_grad():
#         outputs = model(**inputs)

#     return outputs, processor, device


def detect_and_filter_circles(image_path, boxes, centers, edges):
    """
    Detects circles in the image (cans).
    Filters the circles based on the bounding boxes (returned by the object detection model) provided.
    GroundingDino was initially used to detect the bounding boxes of the cans, but it did not meet the accuracy requirements.
    Thus, we only use OpenCV's HoughCircles to detect the circles in the image for now.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blurred_img = cv2.GaussianBlur(img, (0, 0), 1)

    circles = cv2.HoughCircles(blurred_img, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1,
                               minDist=30,
                               param1=200,
                               param2=18,
                               minRadius=10,
                               maxRadius=15)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("Detected circle centers (x, y) and radius:")
        for i in circles[0, :]:
            center = (round(i[0]), round(i[1]))
            radius = i[2]
            # Check if center lies within any of the bounding boxes
            # inside_box = any(
            #     xmin <= center[0] <= xmax and ymin <= center[1] <= ymax
            #     for (xmin, ymin, xmax, ymax) in boxes
            # )
            # if inside_box:
            if True: # For now, we assume all circles are valid (due to the accuracy of the bounding boxes)
                print(f"Center: {center}, Radius: {radius}")
                theta = math.radians(0)
                x = int(center[0] + radius * math.cos(theta))
                y = int(center[1] + radius * math.sin(theta))
                cv2.circle(color, center, 1, (255, 0, 255), 1)
                print(f"Point on circle at angle 0 degrees: ({x-2}, {y})") # -2 for more robustness
                # Store the points for further processing
                centers.append(center)
                edges.append((x-2, y))
            else:
                print(f"Circle at {center} is outside bounding boxes, discarded.")
        return color
    else:
        print("No circles detected.")
        return None
    

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
    # print(f"Pixel ({u}, {v}) to camera coordinates: ({X}, {Y}, {Z})")
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

scipy_quat = R.from_euler("xyz", [0.0, np.pi / 2, 0]).as_quat()
quat_cam = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

_marker_counter = 0

def mark_coordinate(scene, pose, color, radius=0.01):
    """
    Create a visual marker at the specified pose in the scene.
    Args:
        scene (sapien.Scene): The scene to add the marker to.
        pose (list or np.ndarray): The position of the marker in the form [x, y, z].
        color (list): The color of the marker in RGBA format, e.g., [1, 0, 0, 1] for red.
        radius (float): The radius of the marker sphere.
    """
    global _marker_counter
    name = f"marker_{_marker_counter}"
    _marker_counter += 1

    pose = sapien.Pose(p=[pose[0], pose[1], pose[2]])

    can_marker = actors.build_sphere(
        scene=scene,
        radius=radius,
        color=color,
        name=name,
        body_type="kinematic",
        add_collision=False,
        initial_pose=pose
    )
    return can_marker

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
    seed = random.randint(1, 100)
    num_trajs = 0
    print(f"Starting with seed {seed} and {num_trajs} trajectories collected so far.")
    env.reset(seed=seed)

    # Create a directory to save the files
    os.makedirs(f"/home/jypaulsung/Sapien/database/{seed}", exist_ok=True)

    # Capture a screenshot of the scene
    env.unwrapped.scene.update_render()
    camera = env.unwrapped._sensors["base_camera"]
    camera.capture()
    obs_dict = camera.get_obs(rgb=True, depth=True)
    images = camera.get_images(obs_dict)

    # Save the RGB image
    rgb_image = images["rgb"]
    rgb_np = rgb_image.squeeze().cpu().numpy()
    img = Image.fromarray(rgb_np)
    source_path = f"/home/jypaulsung/Sapien/database/{seed}/only_coke_{seed}.png"
    img.save(source_path)

    print(f"Saved RGB screenshot of the cans to {source_path}")

    # Wait for the image to be saved
    while not os.path.exists(source_path):
        print("Waiting for the screenshot to be saved...")
        time.sleep(1) # Wait for 1 second before checking again
    
    time.sleep(1) # Give some time for the file to be fully written

    # Continue processing the image
    # image_path = source_path
    # image = Image.open(image_path).convert("RGB")
    # text = "coke can"
    # outputs, processor, device = run_inference(image, text)

    # width, height = image.size
    # postprocessed_outputs = processor.image_processor.post_process_object_detection(
    #     outputs,
    #     target_sizes=[(height, width)],
    #     threshold=0.25
    # )
    # results = postprocessed_outputs[0]

    boxes = []
    circle_centers = []
    circle_edges = []
    # for score, box in zip(results['scores'], results['boxes']):
    #     xmin, ymin, xmax, ymax = box.tolist()
    #     boxes.append((xmin, ymin, xmax, ymax))    
    
    # plot_results(image, results['scores'].tolist(), results['labels'].tolist(),
    #              results['boxes'].tolist(), text, f"/home/jypaulsung/Sapien/database/{seed}/can_detection_{seed}.png")
    
    cv2.imwrite(f'/home/jypaulsung/Sapien/database/{seed}/detected_circles_{seed}.png', detect_and_filter_circles(source_path, boxes, circle_centers, circle_edges))

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
    for (u, v) in circle_edges:
        u_int, v_int = int(round(u)), int(round(v))
        Z = depth_map[v_int, u_int].cpu().item()
        Z = Z / 1000.0 # convert to meters
        p_cam = pixel_to_camera_coords(u, v, Z, K)
        p_world = camera_to_world(p_cam, extrinsic)
        p_world[1] += 0.019 # add a small offset to the y-coordinate
        world_coords.append(p_world)

    # Check the length of world_coords
    if len(world_coords) != env.unwrapped.num_cans:
        print(f"Warning: Expected {env.unwrapped.num_cans} world coordinates, but got {len(world_coords)}. Can detection might be inaccurate.")
    else:
        print(f"Detected {len(world_coords)} world coordinates for {env.unwrapped.num_cans} cans.")


    # Print the world coordinates (set z to 10.5 before printing)
    for i, coord in enumerate(world_coords):
        coord[2] = 0.105
        print(f"Coke can {i+1} world coordinate: [{coord[0]}, {coord[1]}, {coord[2]}]")        
    
    # Mark the detected can positions in the scene
    for coord in world_coords:
        # Ensure coord is a 1D NumPy array with 3 elements
        if isinstance(coord, torch.Tensor):
            position = coord.squeeze().cpu().numpy()[:3].astype(np.float32)
        else:
            position = np.array(coord[:3], dtype=np.float32)
        
        # Create the Pose with the corrected position
        pose = sapien.Pose(p=position)
        
        # Add the visual sphere markers
        mark_coordinate(env.unwrapped.scene, pose=pose.p, color=[0, 0, 1, 1]) # blue color for the markers

    # Mark the can positions defined in the environment and add them to a list
    starting_reference = []
    for can_pose in env.unwrapped.get_can_poses():
        if isinstance(can_pose, torch.Tensor):
            position = can_pose.squeeze().cpu().numpy()[:3].astype(np.float32)
        else:
            position = np.array(can_pose[:3], dtype=np.float32)
        
        pose = sapien.Pose(p=position)
        starting_reference.append(pose.p)
        
        mark_coordinate(env.unwrapped.scene, pose=pose.p, color=[1, 1, 0, 1]) # yellow color for the markers

    # Save the coordinates in JSON-like .txt file
    detection_data = {
        "circle_centers": [{"x": float(x), "y": float(y)} for x, y in circle_centers],
        "circle_edges_0_deg": [{"x": float(x), "y": float(y)} for x, y in circle_edges],
        "starting_coordinates": [{"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])} for coord in world_coords],
        "starting_reference": [{"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])} for coord in starting_reference]
    }

    
    save_path = f"/home/jypaulsung/Sapien/database/{seed}/can_data_{seed}.txt"
    with open(save_path, "w") as f:
        json.dump(detection_data, f, indent=4)

    print(f"Saved detection data to {save_path}.")

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
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
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
