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
        # Normalize positions to ~[-1,1] for track ~ +/- 50m
        low = np.array([-2]*3 + [-20]*3 + [-np.pi] + [0]*5)
        high = np.array([2]*3 + [20]*3 + [np.pi] + [100]*5)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        self.total_laps = total_laps
        self.current_lap = 0
        self.prev_angle = 0.0
        self.cum_angle = 0.0
        self.steps = 0
        self.max_steps = 5000
        
        # Track parameters (oval center at 0,0)
        self.track_a = 30.0  # semi-major x
        self.track_b = 40.0  # semi-minor y
        
        # Force scales
        self.drive_scale = 1500.0
        self.steer_scale = 300.0
        self.last_steer = 0.0
        
    def step(self, action):
        drive, steer = np.clip(action, -1.0, 1.0)
        car_body_id = self.model.body('car').id
        
        # Apply simple forces
        forward_force = drive * self.drive_scale
        yaw_torque = steer * self.steer_scale
        
        # Apply force in car's forward direction
        quat = np.array(self.data.qpos[3:7]).reshape(4, 1)
        # Build rotation matrix from quaternion (row-major)
        mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(mat.ravel(), quat.ravel())
        forward = mat[:, 0]
        force = forward * forward_force
        self.data.xfrc_applied[car_body_id, :3] = force
        
        # Apply yaw torque around z
        self.data.xfrc_applied[car_body_id, 5] = yaw_torque
        self.last_steer = steer
        
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        # Compute observation
        obs = self._get_obs()
        
        # Reward
        reward = self._get_reward()
        
        # Termination checks
        done = self.current_lap >= self.total_laps or self.steps >= self.max_steps
        truncated = False
        info = {'laps': self.current_lap}
        
        # Off-track penalty / termination
        pos_xy = self.data.qpos[0:2]
        dist_norm = np.sqrt((pos_xy[0]/self.track_a)**2 + (pos_xy[1]/self.track_b)**2)
        dist_to_center = abs(dist_norm - 1.0)
        if dist_norm > 1.2 or self.data.qpos[2] < 0.1:
            reward -= 50.0
            done = True
        
        # Clear applied forces for next step
        self.data.xfrc_applied[car_body_id, :] = 0.0
        
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Start position near bottom of oval with slight random yaw
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = -35.0
        self.data.qpos[2] = 0.5
        yaw = self.np_random.uniform(-0.2, 0.2)
        # Set quaternion from yaw-only (z-rotation)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        self.data.qpos[3:7] = np.array([cy, 0.0, 0.0, sy])
        mujoco.mj_forward(self.model, self.data)
        
        self.current_lap = 0
        self.prev_angle = math.atan2(self.data.qpos[1], self.data.qpos[0])
        self.cum_angle = 0.0
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        pos = self.data.qpos[0:3] / np.array([50.0, 50.0, 5.0])  # normalize
        vel = self.data.qvel[0:3]
        # Simple yaw angle from quaternion (clamp to avoid nan)
        q = self.data.qpos[3:7]
        qw = np.clip(q[0], -1.0, 1.0)
        angle = 2 * np.arccos(qw) * np.sign(q[3])
        
        # Lidar rays (5 rays)
        lidars = []
        ray_angles = np.deg2rad([-60, -30, 0, 30, 60])
        pnt = np.array(self.data.qpos[0:3]).reshape(3, 1)
        geomid = np.array([-1], dtype=np.int32).reshape(1, 1)
        for da in ray_angles:
            dir_vec = np.array([np.cos(angle + da), np.sin(angle + da), 0.0]).reshape(3, 1)
            dist = mujoco.mj_ray(self.model, self.data, pnt, dir_vec, None, 1, -1, geomid)
            # Cap and normalize distance roughly to [0,100]
            lidars.append(min(dist, 100.0) if dist >= 0 else 100.0)
        
        return np.concatenate([pos, vel, [angle], lidars]).astype(np.float32)
    
    def _get_reward(self):
        # Position and centre distance
        pos_xy = self.data.qpos[0:2]
        dist_norm = np.sqrt((pos_xy[0]/self.track_a)**2 + (pos_xy[1]/self.track_b)**2)
        dist_to_center = abs(dist_norm - 1.0)

        # Forward progress incentive (dot product of velocity with track tangent)
        vel_vec = self.data.cvel[1, 0:2].copy()  # car body (id=1) linear x,y
        # Track tangent at current position (for an ellipse x^2/a^2 + y^2/b^2 =1)
        if np.linalg.norm(vel_vec) > 1e-6:
            tangent = np.array([-pos_xy[1] / (self.track_b ** 2), pos_xy[0] / (self.track_a ** 2)])
            tangent /= (np.linalg.norm(tangent) + 1e-8)
            forward_speed = np.dot(vel_vec, tangent)
        else:
            forward_speed = 0.0
        reward = forward_speed  # 1.0 multiplier
        
        # Centerline penalty (quadratic)
        reward += -10.0 * (dist_to_center ** 2)
        
        # Steering effort penalty (encourage smooth driving)
        reward += -0.01 * (self.last_steer ** 2)
        
        # Lap bonus handled below

        # Lap detection via cumulative angle
        current_angle = math.atan2(pos_xy[1], pos_xy[0])
        delta = current_angle - self.prev_angle
        # unwrap
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        self.cum_angle += delta
        self.prev_angle = current_angle
        
        # Count full rotations
        while self.cum_angle >= 2 * math.pi:
            self.cum_angle -= 2 * math.pi
            self.current_lap += 1
            reward += 100.0
        
        # Small alive bonus to stabilize learning
        reward += 0.01
        return float(reward)
    
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
