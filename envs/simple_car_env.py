import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import math
import os

class SimpleCarEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None, total_laps=3):
        super().__init__()
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), '../mj_models/simple_track.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Action: [drive_force, steer]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, angle, 5_lidars]
        low = np.array([-10]*3 + [-5]*3 + [-np.pi] + [0]*5)
        high = np.array([10]*3 + [5]*3 + [np.pi] + [20]*5)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        self.total_laps = total_laps
        self.current_lap = 0
        self.prev_angle = 0.0
        self.steps = 0
        self.max_steps = 2000
        
        # Track parameters (oval center at 0,0)
        self.track_center = np.array([0,0])
        self.track_a = 3.0  # semi-major x
        self.track_b = 4.0  # semi-minor y
        
    def step(self, action):
        drive, steer = action
        car_body_id = self.model.body('car').id
        
        # Apply simple forces
        forward_force = drive * 5.0  # scale force
        yaw_torque = steer * 2.0     # scale torque
        
        # Apply force in car's forward direction
        quat = np.array(self.data.qpos[3:7]).reshape(4, 1)
        mat = np.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(mat.flatten(), quat.flatten())
        forward = mat[:, 0]  # first column is forward vector
        force = forward * forward_force
        self.data.xfrc_applied[car_body_id, :3] = force
        
        # Apply yaw torque
        self.data.xfrc_applied[car_body_id, 5] = yaw_torque  # z-axis torque
        
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        # Compute observation
        obs = self._get_obs()
        
        # Reward
        reward = self._get_reward()
        
        # Done check
        done = self.current_lap >= self.total_laps or self.steps >= self.max_steps
        truncated = False
        info = {'laps': self.current_lap}
        
        # Simple collision detection (height check for off-track)
        if self.data.qpos[2] < 0.1:  # fallen off
            reward -= 50
            done = True
        
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # Start position
        self.data.qpos[0] = 0
        self.data.qpos[1] = -3
        self.data.qpos[2] = 0.2
        mujoco.mj_forward(self.model, self.data)
        
        self.current_lap = 0
        self.prev_angle = 0.0
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        # Simple yaw angle from quaternion (clamp to avoid nan)
        q = self.data.qpos[3:7]
        qw = np.clip(q[0], -1.0, 1.0)  # clamp for arccos
        angle = 2 * np.arccos(qw) * np.sign(q[3])  # approximate yaw
        
        # Lidar rays (5 rays)
        lidars = []
        ray_angles = np.deg2rad([-60, -30, 0, 30, 60])
        pnt = np.array(pos).reshape(3, 1)  # shape (3,1)
        geomid = np.array([-1], dtype=np.int32).reshape(1, 1)  # writable (1,1)
        for da in ray_angles:
            dir_vec = np.array([np.cos(angle + da), np.sin(angle + da), 0]).reshape(3, 1)
            dist = mujoco.mj_ray(self.model, self.data, pnt, dir_vec, None, 1, -1, geomid)
            lidars.append(min(dist, 20.0) if dist >= 0 else 20.0)
        
        return np.concatenate([pos, vel, [angle], lidars])
    
    def _get_reward(self):
        pos = self.data.qpos[0:2]
        vel = np.linalg.norm(self.data.qvel[0:2])
        
        # Progress: distance to center line
        dist_to_center = np.abs(np.sqrt((pos[0]/self.track_a)**2 + (pos[1]/self.track_b)**2) - 1)
        progress_reward = vel * (1 - dist_to_center)
        
        # Lap detection
        current_angle = np.arctan2(pos[1], pos[0])
        if current_angle - self.prev_angle > np.pi:
            self.current_lap += 1
            progress_reward += 100  # lap bonus
        self.prev_angle = current_angle
        
        # Off-track penalty
        if dist_to_center > 0.5:
            progress_reward -= 10
        
        return progress_reward + 0.1  # small alive bonus
    
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            renderer = mujoco.Renderer(self.model)
            renderer.update_scene(self.data, camera='follow_cam')
            return renderer.render()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
