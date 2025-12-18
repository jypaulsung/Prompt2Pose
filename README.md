# ArrayCan
Defining a task arraying can with SAPIEN simulator

![demo](https://github.com/user-attachments/assets/a4a5964a-90f7-47b2-9e2f-17379d1217a9)

## Environment Used
* Ubuntu 22.04 / 20.04
* SAPIEN 3.x
* mani skill 3.x
* mplib 0.2.x
* Panda

[ManiSkill User Guide](https://maniskill.readthedocs.io/en/v3.0.0b20/user_guide/index.html)  
[MPlib Github](https://github.com/haosulab/MPlib)  

## Structure of package of mani_skill
```sh
.
mani_skill
├── xxx
├── xxx
├── xxx
├── xxx
├── assets
│   ├── ...
│   └── models
│     └── coke
│     └── pepsi
├── examples
│   ├── xxx
│   ├── xxx
│   ├── xxx
│   └── xxx
└── envs
     └── tasks
         └── tabletop
            └── array_can.py
```
## Set up
```
conda env create -f sapien3.yml
pip install mani_skill
pip install mplib==0.2
```
Move the files included to the instructed directories
@mani_skill/envs/tasks/tabletop/__init__.py
```
+) from .array_can import ArrayCanEnv
```
@mani_skill/examples/motionplanning/panda/motionplanner.py
```
# at line81
-) planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))

+) from mplib.pymp import Pose as MPPose
arr = np.hstack([self.base_pose.p, self.base_pose.q]).astype(np.float64).ravel()
        
        p = arr[:3].reshape(3, 1)
        q = arr[3:].reshape(4, 1)
        
        pose = MPPose(p=p, q=q)
        planner.set_base_pose(pose)
# edit move_to_pose_with_screw function
    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        # pose = sapien.Pose(p=pose.p , q=pose.q)
        pose = MPPose(pose.p, pose.q)
        result = self.planner.plan_screw(
            # np.concatenate([pose.p, pose.q]),
            pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                # np.concatenate([pose.p, pose.q]),
                pose,
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)
```
## Usage

You can test the installation with
```
python -m mani_skill.examples.teleoperation.interactive_panda -e ArrayCan
```
