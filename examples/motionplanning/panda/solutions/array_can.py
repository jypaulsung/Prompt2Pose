import numpy as np
import sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult
from mani_skill.envs.tasks import ArrayCanEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


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

    # 모든 캔 위치 (나중에 VLM으로 대체)
    # seed = 0
    pick = [[0.11796,  0.056426, 0.08],
            [0.0148988,0.205559, 0.08],
            [-0.0594433,0.199338,0.08],
            [-0.0628068,0.0752942,0.08],
            [-0.12443, -0.0406532,0.08]]
    # # seed = 1
    # pick = ([0.066385, 0.0737458, 0.08], 
    #         [0.037905, 0.0130262, 0.08], 
    #         [0.143296, -0.0365266, 0.08], 
    #         [0.121768, -0.115834, 0.08], 
    #         [0.0418419, -0.12208, 0.08])
    # # seed = 2
    # pick = ([-0.00316405, 0.0917179, 0.08], 
    #         [-0.0914487, 0.0033105, 0.08], 
    #         [0.0155603, 0.00856102, 0.08], 
    #         [-0.058669, -0.0886018, 0.08], 
    #         [0.070955, -0.0385983, 0.08])

    place = [[-0.24,0.28,0.08],
             [-0.24,0.18,0.08],
             [-0.24,0.08,0.08],
             [-0.24,-0.02,0.08],
             [-0.24,-0.12,0.08]]
    
    for i in range(env.num_cans):
        pick = pick_place_with_obstacles(env.unwrapped, solver, i, pick[i], place[i], pick)


    solver.close()
    return True
