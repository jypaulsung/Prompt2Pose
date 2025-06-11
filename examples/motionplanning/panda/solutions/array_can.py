import numpy as np
import sapien
import json
from pathlib import Path
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult
from mani_skill.envs.tasks import ArrayCanEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# def load_world_coordinates(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data = data.get('iter0', [])
#     return data.get('starting_coordinates', [])
def load_world_coordinates(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def reorder_pick_place(can_extent, pick_list, place_list):
    """
    can_extent: [x_extent, y_extent, z_extent]
    pick_list: list of [x, y, z]
    place_list: list of [x, y, z]
    """
    picks = np.array(pick_list)
    places = np.array(place_list)
    r = can_extent[0]
    
    num_picks = len(picks)
    num_places = len(places)
    
    distances = np.linalg.norm(picks[:, None, :2] - places[None, :, :2], axis=2)
    
    matches = []
    for i in range(num_picks):
        for j in range(num_places):
            d = distances[i, j]
            if d <= r:
                matches.append((i, j, d))
                
    matches_sorted = sorted(matches, key=lambda x: x[2])
    used_picks = set()
    used_places = set()
    matched_pairs = []
    for i, j, d in matches_sorted:
        if i not in used_picks and j not in used_places:
            matched_pairs.append((i, j))
            used_picks.add(i)
            used_places.add(j)
    
    matched_picks = [i for i, _ in matched_pairs]
    matched_places = [j for _, j in matched_pairs]
    
    remaining_picks = [i for i in range(num_picks) if i not in used_picks]
    remaining_places = [j for j in range(num_places) if j not in used_places]
    
    for i, j in zip(remaining_picks, remaining_places):
        matched_picks.append(i)
        matched_places.append(j)
    
    new_picks = [pick_list[i] for i in matched_picks]
    new_places = [place_list[j] for j in matched_places]
    
    return new_picks, new_places

def move_with_rotation_retries(planner, pose: sapien.Pose, max_retries=12):
    """
    planner.move_to_pose_with_screw(pose) 가 실패하면
    Z축으로 30°씩 누적 회전시키며 재시도.
    """
    base_q = np.array(pose.q, dtype=float)

    attempt = 0
    while attempt < max_retries:
        ret = planner.move_to_pose_with_screw(pose)
        if ret != -1:
            return ret
        yaw = np.deg2rad(30 * (attempt + 1))
        w, x, y, z = euler2quat(0, 0, yaw)
        q_rot = np.array([w, x, y, z], dtype=float)
        q_new = qmult(q_rot, base_q)
        pose.q = q_new.tolist()

        attempt += 1
    
    raise RuntimeError(f"screw plan failed after {max_retries} retries")
    

def pick_place_with_obstacles(
    env,
    planner: PandaArmMotionPlanningSolver,
    index: int,
    pick: list[float] | np.ndarray,
    place: list[float] | np.ndarray,
    all_cans: list[list[float]],
    lift_height: float = 0.15,
    can_extent: np.ndarray = np.array([0.06, 0.06, 0.105]),
):
    pick = np.array(pick, dtype=float)
    place = np.array(place, dtype=float)

    # add collisions
    planner.clear_collisions()
    for can_pos in all_cans:
        if np.allclose(can_pos, pick[:3], atol=1e-3):
            continue
        center = [can_pos[0], can_pos[1], can_extent[2] / 2]
        pose = sapien.Pose(p=center, q=[0,0,0,1])
        planner.add_box_collision(can_extent, pose)

    # base quaternion
    current_q = env.unwrapped.agent.tcp.pose.q[0]
    q_pick  = pick[3:] if pick.size == 7 else current_q
    q_place = place[3:] if place.size == 7 else current_q

    # pick
    grasp_pose = sapien.Pose(p=pick[:3].tolist(), q=q_pick.tolist())
    lift_pose   = sapien.Pose([0,0,lift_height]) * grasp_pose

    # approach -> grasp -> lift
    move_with_rotation_retries(planner, lift_pose)
    move_with_rotation_retries(planner, grasp_pose)
    planner.close_gripper()
    move_with_rotation_retries(planner, lift_pose)

    # place
    place_pose  = sapien.Pose(p=place[:3].tolist(), q=q_place.tolist())
    above_pose  = sapien.Pose([0,0,lift_height]) * place_pose

    move_with_rotation_retries(planner, above_pose)
    move_with_rotation_retries(planner, place_pose)
    planner.open_gripper()
    move_with_rotation_retries(planner, above_pose)
    
    all_cans[index] = place
    return all_cans

def solve(env: ArrayCanEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    solver = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    file = Path(__file__).resolve()
    project_root = file.parents[4]
    path = f"{project_root}/dataset/{seed}/processed_can_data_{seed}.txt"
    data = load_world_coordinates(path)
    world_coords = data.get('iter0', [])
    world_coords = world_coords.get('starting_coordinates', [])
    pick = []
    for i, coord in enumerate(world_coords, start=1):
        pick.append([coord['x'], coord['y'], 0.08])
    
    dest_coords = data.get('destination_coordinates', [])
    place = []
    for i, coord in enumerate(dest_coords, start=1):
        place.append([coord['x'], coord['y'], 0.08])

    pick, place = reorder_pick_place([0.06, 0.06, 0.105], pick, place)
    
    for i in range(env.num_cans):
        pick = pick_place_with_obstacles(env.unwrapped, solver, i, pick[i], place[i], pick)

    poses = env.get_can_poses()
    coords_list = []
    for t in poses:
        coords_list.append(t.squeeze().tolist())
    
    path = Path(f"{project_root}/dataset/{seed}/can_dest_{seed}.txt")
    
    if path.exists():
        with open(path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    existing_iters = [int(k.replace("iter", "")) for k in all_data.keys() if k.startswith("iter")]
    next_iter = max(existing_iters, default=-1) + 1
    iter_key = f"iter{next_iter}"

    all_data[iter_key] = coords_list

    with open(path, "w") as f:
        json.dump(all_data, f, indent=4)


    print(f"Saved detection data to {path}.")
    res = solver.open_gripper()

    solver.close()
    return res
