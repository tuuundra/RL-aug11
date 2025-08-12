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
        
        # Observation: [norm_pos_x, norm_pos_y, cos_angle, sin_angle, norm_vel, norm_lidars*5, norm_lap, next_curve_angle]
        low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0] + [0.0] * 5 + [0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 5 + [1.0, 1.0])
        self.observation_space = spaces.Box(low=low, high=high, shape=(12,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        self.total_laps = total_laps
        self.current_lap = 0
        self.prev_angle = 0.0
        self.cum_angle = 0.0
        self.steps = 0
        self.max_steps = 5000
        
        # Track parameters (oval center at 0,0, scaled to match car-RL proportions)
        self.track_a = 60.0  # semi-major x (doubled for MuJoCo scale)
        self.track_b = 30.0  # semi-minor y
        self.track_width = 8.0  # Adjusted for car size
        
        # Generate waypoints on centerline
        self.waypoints = []
        num_points = 36  # every 10 degrees
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = self.track_a * math.cos(theta)
            y = self.track_b * math.sin(theta)
            self.waypoints.append((x, y))
        
        self.current_waypoint = 0
        self.max_vel = 20.0
        self.min_vel_threshold = 1.0
        self.max_lidar_dist = 100.0
        
        # Force scales tuned for 2-D planar cart
        self.drive_scale = 1500.0
        self.steer_scale = 150.0
        self.last_steer = 0.0
        
    def _ellipse_eq(self, x, y, a, b):
        return ((x / a) ** 2 + (y / b) ** 2)
    
    def _is_on_track(self, x, y):
        min_dist = float('inf')
        for wx, wy in self.waypoints:
            dist = math.sqrt((x - wx)**2 + (y - wy)**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist <= self.track_width / 2.0
    
    def _dist_to_centerline(self):
        min_dist = float('inf')
        pos_xy = self.data.qpos[0:2]
        for wx, wy in self.waypoints:
            dist = math.sqrt((pos_xy[0] - wx)**2 + (pos_xy[1] - wy)**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _update_waypoint(self):
        min_dist = float('inf')
        pos_xy = self.data.qpos[0:2]
        closest_idx = 0
        for i, (wx, wy) in enumerate(self.waypoints):
            dist = math.sqrt((pos_xy[0] - wx)**2 + (pos_xy[1] - wy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx > self.current_waypoint or (closest_idx == 0 and self.current_waypoint > len(self.waypoints) // 2):
            self.current_waypoint = closest_idx
    
    def _is_lap_complete(self):
        pos_xy = self.data.qpos[0:2]
        if self.current_waypoint >= len(self.waypoints) - 1 and math.sqrt((pos_xy[0] - self.waypoints[0][0])**2 + (pos_xy[1] - self.waypoints[0][1])**2) < 5.0:
            self.current_lap += 1
            self.current_waypoint = 0
            return True
        return False
        
    def step(self, action):
        drive, steer = np.clip(action, -1.0, 1.0)
        car_body_id = self.model.body('car').id
        
        # Apply simple forces
        forward_force = drive * self.drive_scale
        yaw_torque = steer * self.steer_scale
        
        yaw = self.data.qpos[2]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.data.xfrc_applied[car_body_id, :3] = forward_dir * forward_force
        
        # Apply yaw torque around z
        self.data.xfrc_applied[car_body_id, 5] = yaw_torque
        self.last_steer = steer
        
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        self._update_waypoint()
        
        # Compute observation
        obs = self._get_obs()
        
        # Reward (pass action for penalty)
        reward = self._get_reward(action)
        
        # Termination checks
        terminated = self.current_lap >= self.total_laps or self.steps >= self.max_steps or not self._is_on_track(self.data.qpos[0], self.data.qpos[1])
        truncated = False
        info = {'laps': self.current_lap}
        
        # Clear applied forces for next step
        self.data.xfrc_applied[car_body_id, :] = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Perturb waypoints for variety
        original_waypoints = [(self.track_a * math.cos(2 * math.pi * i / 36),
                               self.track_b * math.sin(2 * math.pi * i / 36))
                              for i in range(36)]

        # Keep variety for the path but spawn exactly on the centre-line
        self.waypoints = [(x + self.np_random.uniform(-3, 3), y + self.np_random.uniform(-3, 3))
                          for x, y in original_waypoints]

        # Start at bottom centre of oval
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = -self.track_b
        self.data.qpos[2] = 0.0  # yaw
        # Small random yaw perturbation around forward (+x)
        self.data.qpos[2] = self.np_random.uniform(-0.1, 0.1)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_lap = 0
        self.current_waypoint = 0
        self.steps = 0
        self.prev_angle = math.atan2(self.data.qpos[1], self.data.qpos[0])
        self.cum_angle = 0.0
        
        return self._get_obs(), {}
    
    def _update_lidar_visuals(self, ray_dirs: np.ndarray, ray_dists: np.ndarray) -> None:
        """Update green capsule geoms to match lidar rays (visual only)."""
        # Car origin in world
        car_pos = self.data.qpos[0:3].copy()
        ray_dirs = np.asarray(ray_dirs).reshape(-1, 3)
        ray_dists = np.asarray(ray_dists).reshape(-1)
        z_axis = np.array([0.0, 0.0, 1.0])
        quat = np.zeros(4)
        for i in range(5):
            name = f"lidar_ray_{i}"
            try:
                gid = self.model.geom(name).id
            except Exception:
                continue
            length = float(max(ray_dists[i], 1e-6))
            direction = ray_dirs[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = z_axis
            else:
                direction = direction / norm
            # Compute quaternion rotating z-axis to direction
            c = float(np.dot(z_axis, direction))
            if c > 0.9999:
                # Aligned
                quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
            elif c < -0.9999:
                # Opposite
                axis = np.array([1.0, 0.0, 0.0])
                mujoco.mju_axisAngle2Quat(quat, axis, np.pi)
            else:
                axis = np.cross(z_axis, direction)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                angle = float(np.arccos(np.clip(c, -1.0, 1.0)))
                mujoco.mju_axisAngle2Quat(quat, axis, angle)
            # Center point
            center = car_pos + 0.5 * direction * length
            # Update geom pose and size (capsule size: [radius, half-length])
            self.model.geom_pos[gid] = center
            self.model.geom_quat[gid] = quat
            self.model.geom_size[gid, 0] = 0.05  # radius
            self.model.geom_size[gid, 1] = length / 2.0  # half-length
        
    def _get_obs(self):
        pos = self.data.qpos[0:2] / np.array([self.track_a + 10, self.track_b + 10]) * 2 - 1  # Normalize to [-1,1]
        
        # Angle as cos/sin
        yaw = self.data.qpos[2]
        cos_angle = np.cos(yaw)
        sin_angle = np.sin(yaw)
        
        # Normalized speed (scalar)
        vel = np.linalg.norm(self.data.qvel[0:2]) / self.max_vel
        
        # Lidar rays normalized to [0,1]
        lidars = []
        ray_angles = np.deg2rad([-60, -30, 0, 30, 60])
        pnt = np.array(self.data.qpos[0:3]).reshape(3, 1)
        geomid = np.array([-1], dtype=np.int32).reshape(1, 1)
        ray_dirs = []
        ray_dists = []
        car_pos = self.data.qpos[0:3].copy()
        # Rotation matrix from car body quaternion
        yaw = self.data.qpos[2]
        body_mat = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                             [-np.sin(yaw), np.cos(yaw), 0],
                             [          0,           0, 1]])
        for da in ray_angles:
            dir_vec = np.array([np.cos(yaw + da), np.sin(yaw + da), 0.0]).reshape(3, 1)
            dist = mujoco.mj_ray(self.model, self.data, pnt, dir_vec, None, 1, -1, geomid)
            capped = min(max(dist, 0), self.max_lidar_dist)
            lidars.append(capped / self.max_lidar_dist)
            ray_dirs.append(dir_vec.ravel())
            ray_dists.append(capped)

        # Update lidar tip site positions (local frame)
        for i, (dvec, dlen) in enumerate(zip(ray_dirs, ray_dists)):
            tip_id = self.model.site(f'lidar_tip_{i}').id
            world_end = car_pos + dvec * dlen
            local_end = body_mat.T @ (world_end - car_pos)
            self.model.site_pos[tip_id] = local_end
        
        # Update visualization
        try:
            self._update_lidar_visuals(np.array(ray_dirs), np.array(ray_dists))
        except Exception:
            pass
        
        norm_lap = self.current_lap / self.total_laps
        
        next_curve_angle = 0.0
        if self.current_waypoint + 2 < len(self.waypoints):
            next_idx = (self.current_waypoint + 1) % len(self.waypoints)
            next_next_idx = (next_idx + 1) % len(self.waypoints)
            vec1 = np.array(self.waypoints[next_idx]) - np.array(self.waypoints[self.current_waypoint])
            vec2 = np.array(self.waypoints[next_next_idx]) - np.array(self.waypoints[next_idx])
            next_curve_angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8), -1.0, 1.0)) / np.pi  # [0,1]
        
        return np.array([pos[0], pos[1], cos_angle, sin_angle, vel] + lidars + [norm_lap, next_curve_angle]).astype(np.float32)
    
    def _get_reward(self, action):
        pos_xy = self.data.qpos[0:2]
        car_angle = self.data.qpos[2]  # yaw only
        car_vel = np.linalg.norm(self.data.qvel[0:2])
        drive, steer = np.clip(action, -1.0, 1.0)  # Assuming action available; if not, pass from step
        
        reward = 0.0
        
        if not self._is_on_track(pos_xy[0], pos_xy[1]):
            reward = -100.0
        else:
            dist_to_center = self._dist_to_centerline()
            next_idx = (self.current_waypoint + 1) % len(self.waypoints)
            dir_to_next = np.array([self.waypoints[next_idx][0] - pos_xy[0], self.waypoints[next_idx][1] - pos_xy[1]])
            norm = np.linalg.norm(dir_to_next)
            if norm > 0:
                unit_dir = dir_to_next / norm
                vel_vec = np.array([np.cos(car_angle) * car_vel, np.sin(car_angle) * car_vel])
                progress = np.dot(vel_vec, unit_dir)
                velocity_bonus = (car_vel / self.max_vel) * 0.5
                stillness_penalty = -0.5 * (1 - car_vel / self.min_vel_threshold) if car_vel < self.min_vel_threshold else 0.0
                action_penalty = -0.01 * (abs(steer) + abs(drive))
                
                # Boundary proximity penalty (using lidars from obs or recompute)
                lidars = self._get_obs()[5:10].tolist()  # Extract from obs
                min_lidar = min(lidars) * self.max_lidar_dist  # Unnormalize
                boundary_threshold = 10.0
                boundary_penalty = -0.5 * (boundary_threshold - min_lidar) if min_lidar < boundary_threshold else 0.0
                
                # Curve-aware speed penalty
                curve_penalty = 0.0
                if self.current_waypoint + 1 < len(self.waypoints):
                    next_idx = (self.current_waypoint + 1) % len(self.waypoints)
                    next_next_idx = (next_idx + 1) % len(self.waypoints)
                    vec1 = np.array(self.waypoints[next_idx]) - np.array(self.waypoints[self.current_waypoint])
                    vec2 = np.array(self.waypoints[next_next_idx]) - np.array(self.waypoints[next_idx])
                    angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8), -1.0, 1.0))
                    if angle > np.radians(45) and car_vel > 10.0:
                        curve_penalty = -0.2 * (car_vel - 10.0)
                
                # Anticipation penalty
                anticipation_penalty = 0.0
                if self.current_waypoint + 2 < len(self.waypoints):
                    next_idx = (self.current_waypoint + 1) % len(self.waypoints)
                    next_next_idx = (next_idx + 1) % len(self.waypoints)
                    vec1 = np.array(self.waypoints[next_idx]) - np.array(self.waypoints[self.current_waypoint])
                    vec2 = np.array(self.waypoints[next_next_idx]) - np.array(self.waypoints[next_idx])
                    angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8), -1.0, 1.0))
                    if angle > np.radians(45) and car_vel > 8.0:
                        anticipation_penalty = -0.1 * (car_vel - 8.0)
                
                # Boundary distance normalization penalty
                avg_lidar = np.mean(lidars) * self.max_lidar_dist
                optimal_min = 15.0
                optimal_max = 30.0
                distance_penalty = 0.0
                if avg_lidar < optimal_min:
                    distance_penalty = -0.1 * (optimal_min - avg_lidar)
                elif avg_lidar > optimal_max:
                    distance_penalty = -0.1 * (avg_lidar - optimal_max)
                
                reward = (2.0 * progress) - 0.5 * dist_to_center + velocity_bonus + stillness_penalty + action_penalty + boundary_penalty + curve_penalty + anticipation_penalty + distance_penalty + 1.0
            else:
                reward = -1.0
        
        if self._is_lap_complete():
            reward += 300.0 / (self.steps + 1)
        
        return float(reward)
    
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        elif self.render_mode == 'rgb_array':
            renderer = mujoco.Renderer(self.model, height=1080, width=1920)
            renderer.update_scene(self.data, camera='birds_eye')
            return renderer.render()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
