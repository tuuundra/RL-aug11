import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import math
import os

class SimpleCarEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None, total_laps=3, eval_mode=False):
        super().__init__()
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), '../mj_models/simple_track.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Action: [drive_force, steer]. Disallow reverse; throttle in [0,1]
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation: [norm_pos_x, norm_pos_y, cos_angle, sin_angle, norm_vel, norm_lidars*5, norm_lap, next_curve_angle]
        low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0] + [0.0] * 5 + [0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 5 + [1.0, 1.0])
        self.observation_space = spaces.Box(low=low, high=high, shape=(12,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        self.total_laps = total_laps
        self.eval_mode = eval_mode
        self.current_lap = 0
        self.prev_angle = 0.0
        self.cum_angle = 0.0
        self.steps = 0
        self.max_steps = 5000
        
        # Lap completion tracking
        self.last_lap_step = 0  # Step count when last lap was completed
        self.min_steps_per_lap = 300  # Minimum steps required between laps
        
        # Track parameters (oval center at 0,0, scaled to match car-RL proportions)
        self.track_a = 60.0  # semi-major x (doubled for MuJoCo scale)
        self.track_b = 30.0  # semi-minor y
        self.track_width = 12.0  # Moderated: wider 12 m lane for tolerance
        
        # Fixed boundary ellipses (like Pygame version - NEVER change)
        self.a_outer = self.track_a + self.track_width / 2.0  # 64.0
        self.b_outer = self.track_b + self.track_width / 2.0  # 34.0
        self.a_inner = self.track_a - self.track_width / 2.0  # 56.0
        self.b_inner = self.track_b - self.track_width / 2.0  # 26.0
        
        # Generate waypoints on centerline (CLOCKWISE direction)
        self.waypoints = []
        self.safe_waypoints = []  # CRITICAL FIX: Safe waypoints for progress calculation
        num_points = 36  # every 10 degrees
        for i in range(num_points):
            theta = -2 * math.pi * i / num_points  # NEGATIVE for clockwise
            x = self.track_a * math.cos(theta)
            y = self.track_b * math.sin(theta)
            self.waypoints.append((x, y))
            self.safe_waypoints.append((x, y))  # Keep unperturbed for safety
        
        self.current_waypoint = 0
        self.max_vel = 20.0
        self.min_vel_threshold = 1.0
        self.max_lidar_dist = 40.0   # better scaling: outer wall ~34 m
        
        # Optimized force scales for realistic racing speeds
        self.drive_scale = 400.0   # Moderated: ≈2.7 m/s² acceleration (150 kg)
        self.steer_scale = 120.0   # Restore stronger steering torque
        self.last_steer = 0.0
        
    def _ellipse_eq(self, x, y, a, b):
        """Calculate ellipse equation value for point (x,y) with semi-axes a,b"""
        return (x / a) ** 2 + (y / b) ** 2
    
    def _is_on_track(self, x, y):
        """Check if position is within track boundaries using ellipse math (like Pygame)"""
        eq_outer = self._ellipse_eq(x, y, self.a_outer, self.b_outer)
        eq_inner = self._ellipse_eq(x, y, self.a_inner, self.b_inner)
        return eq_outer <= 1.0 and eq_inner >= 1.0
    
    def _cast_lidar_ray(self, start_x, start_y, dir_x, dir_y):
        """Cast a lidar ray and find distance to track boundary using ellipse intersection"""
        # Step along the ray until we hit a boundary
        step_size = 0.5  # Small steps for accuracy
        dist = 0.0
        
        while dist < self.max_lidar_dist:
            # Current position along ray
            px = start_x + dist * dir_x
            py = start_y + dist * dir_y
            
            # Check if we've hit a boundary (outside track)
            if not self._is_on_track(px, py):
                return dist
            
            dist += step_size
        
        return self.max_lidar_dist  # No boundary hit within max range
    
    def _dist_to_centerline(self):
        min_dist = float('inf')
        pos_xy = self.data.qpos[0:2]
        for wx, wy in self.waypoints:
            dist = math.sqrt((pos_xy[0] - wx)**2 + (pos_xy[1] - wy)**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _update_waypoint(self):
        pos_xy = self.data.qpos[0:2]
        
        # Store previous waypoint for lap detection
        prev_waypoint = self.current_waypoint
        
        # Check if we're close enough to the NEXT waypoint to advance
        next_waypoint_idx = (self.current_waypoint + 1) % len(self.waypoints)
        next_wx, next_wy = self.waypoints[next_waypoint_idx]
        dist_to_next = math.sqrt((pos_xy[0] - next_wx)**2 + (pos_xy[1] - next_wy)**2)
        
        # Advance to next waypoint if we're within threshold distance
        waypoint_threshold = 15.0  # Distance threshold to advance waypoint
        if dist_to_next < waypoint_threshold:
            self.current_waypoint = next_waypoint_idx
            
            # Check for lap completion: waypoint wrapped from high to low (e.g., 35 -> 0)
            if prev_waypoint > len(self.waypoints) // 2 and self.current_waypoint == 0:
                self._check_lap_completion()
    
    def _check_lap_completion(self):
        """Check if a lap is truly complete using proper conditions"""
        # 1. Min steps guard: Must have enough steps since last lap
        steps_since_last_lap = self.steps - self.last_lap_step
        if steps_since_last_lap < self.min_steps_per_lap:
            return False
            
        # 2. Forward crossing check: Ensure moving in correct direction (clockwise)
        pos_xy = self.data.qpos[0:2]
        vel_xy = self.data.qvel[0:2]
        
        # Start line normal vector (pointing inward for clockwise motion)
        # For our track, start line is roughly at x=track_a, y=0
        # Normal should point toward negative x for clockwise crossing
        start_line_normal = np.array([-1.0, 0.0])  # Points inward (left) for clockwise
        
        # Check if velocity has positive component along normal (crossing in correct direction)
        forward_crossing = np.dot(vel_xy, start_line_normal) > 0.5  # Require some minimum speed
        
        if forward_crossing:
            self.current_lap += 1
            self.last_lap_step = self.steps
            return True
        
        return False
    
    def _is_lap_complete(self):
        """Legacy method - now just checks if lap count increased this step"""
        # This will be called from reward function to detect lap completion
        # The actual lap logic is now in _check_lap_completion via _update_waypoint
        return False  # Lap completion is now handled in _update_waypoint
        
    def step(self, action):
        drive, steer = np.clip(action, -1.0, 1.0)
        # Clip drive to [0,1] to avoid reverse motion
        drive = np.clip(drive, 0.0, 1.0)
        car_body_id = self.model.body('car').id
        
        # Apply physics-optimized forces with velocity damping
        forward_force = drive * self.drive_scale
        yaw_torque = steer * self.steer_scale
        
        # Add velocity-dependent air resistance for realism (reduced)
        current_vel = np.linalg.norm(self.data.qvel[:2])
        air_resistance = -0.03 * current_vel * current_vel  # Reduced quadratic air resistance
        
        yaw = self.data.qpos[2]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        total_force = (forward_force + air_resistance)
        self.data.xfrc_applied[car_body_id, :3] = forward_dir * total_force
        
        # Apply yaw torque with velocity-dependent scaling for better control
        speed_factor = min(1.0, current_vel / 10.0)  # Reduce steering at low speeds
        self.data.xfrc_applied[car_body_id, 5] = yaw_torque * (0.3 + 0.7 * speed_factor)
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
        info = {'laps': self.current_lap, 'reward_components': getattr(self, '_last_reward_breakdown', {})}
        
        # Clear applied forces for next step
        self.data.xfrc_applied[car_body_id, :] = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # CRITICAL FIX: Separate safe and variety waypoints
        # Safe waypoints for progress calculation (never perturbed)
        self.safe_waypoints = [(self.track_a * math.cos(-2 * math.pi * i / 36),
                                self.track_b * math.sin(-2 * math.pi * i / 36))
                               for i in range(36)]
        
        # Use fixed safe waypoints to keep the task stationary during training
        self.waypoints = list(self.safe_waypoints)  # No random perturbation

        # Start at first waypoint position (like Pygame version)
        start_waypoint = self.safe_waypoints[0]  # Use safe waypoint for consistent start
        self.data.qpos[0] = start_waypoint[0]
        self.data.qpos[1] = start_waypoint[1]
        
        # Set initial orientation to point toward next waypoint (like Pygame version)
        next_waypoint = self.safe_waypoints[1]
        dx = next_waypoint[0] - start_waypoint[0]
        dy = next_waypoint[1] - start_waypoint[1]
        initial_yaw = math.atan2(dy, dx)
        self.data.qpos[2] = initial_yaw
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_lap = 0
        self.current_waypoint = 0
        self.steps = 0
        self.last_lap_step = 0  # Reset lap tracking
        self._prev_lap_count = 0  # Reset reward lap tracking
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
        
        # Custom lidar using ellipse boundary math (like Pygame version)
        lidars = []
        ray_angles = np.deg2rad([-60, -30, 0, 30, 60])
        ray_dirs = []
        ray_dists = []
        car_pos = self.data.qpos[0:2]  # Only x,y needed
        yaw = self.data.qpos[2]
        
        for da in ray_angles:
            # Calculate ray direction
            ray_yaw = yaw + da
            dir_x = np.cos(ray_yaw)
            dir_y = np.sin(ray_yaw)
            
            # Cast ray using ellipse intersection (like original Pygame lidar)
            dist = self._cast_lidar_ray(car_pos[0], car_pos[1], dir_x, dir_y)
            
            capped = min(max(dist, 0), self.max_lidar_dist)
            lidars.append(capped / self.max_lidar_dist)
            ray_dirs.append(np.array([dir_x, dir_y, 0.0]))
            ray_dists.append(capped)

        # --- NEW: lidar-based curve indicator (left rays minus right rays) ---
        curve_ind = (lidars[0] + lidars[1]) - (lidars[3] + lidars[4])  # raw range approx [-2, 2]
        curve_ind_raw = float(-curve_ind / 2.0)  # flip sign so positive means turn right, scale to [-1,1]
        
        # Update lidar tip site positions (local frame)
        car_pos_3d = self.data.qpos[0:3].copy()  # Use 3D position for visualization
        body_mat = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                             [-np.sin(yaw), np.cos(yaw), 0],
                             [          0,           0, 1]])
        for i, (dvec, dlen) in enumerate(zip(ray_dirs, ray_dists)):
            tip_id = self.model.site(f'lidar_tip_{i}').id
            world_end = car_pos_3d + dvec * dlen
            local_end = body_mat.T @ (world_end - car_pos_3d)
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
        
        # Replace next_curve_angle in observation with sensor-derived curve indicator
        return np.array([pos[0], pos[1], cos_angle, sin_angle, vel] + lidars + [norm_lap, curve_ind_raw]).astype(np.float32)
    
    def _get_reward(self, action):
        """IMPROVED REWARD SYSTEM - Fixed critical issues"""
        pos_xy = self.data.qpos[0:2]
        car_angle = self.data.qpos[2]
        car_vel = np.linalg.norm(self.data.qvel[0:2])
        drive = action[0]  # Throttle value for brake reward
        
        # Off-track termination
        if not self._is_on_track(pos_xy[0], pos_xy[1]):
            return -100.0
        
        # Get lidar readings for safety
        lidars = self._get_obs()[5:10].tolist()
        # Curve indicator from lidar (same formula as in _get_obs)
        curve_ind = (lidars[0] + lidars[1]) - (lidars[3] + lidars[4])
        curve_ind = -curve_ind / 2.0  # sign flipped, scaled
        min_lidar = min(lidars) * self.max_lidar_dist
        
        # 1. FORWARD PROGRESS REWARD (Primary driver) - FIXED: Use same waypoints as tracking
        next_idx = (self.current_waypoint + 1) % len(self.waypoints)
        # Tangent vector along track centre-line (current waypoint -> next waypoint)
        tangent = np.array(self.waypoints[next_idx]) - np.array(self.waypoints[self.current_waypoint])
        tan_norm = np.linalg.norm(tangent)
        if tan_norm > 0:
            unit_tan = tangent / tan_norm
            vel_vec = np.array([np.cos(car_angle) * car_vel, np.sin(car_angle) * car_vel])
            progress = np.dot(vel_vec, unit_tan)  # Positive when moving along track direction
            progress_reward = np.tanh(progress / 10.0)
        else:
            progress_reward = 0.0
        
        # 2. SPEED INCENTIVE AND CURVE-AWARE PENALTY --- simplified
        # --- Simplified reward terms ---
        # Positive speed incentive with idle penalty (reduced cap)
        if car_vel < 1.0:
            speed_reward = -0.2
        else:
            speed_reward = 1.0 * np.tanh(car_vel / 2.0)

        # Curve-aware target speed and penalty (only if already moving fast)
        curve_speed_penalty = 0.0
        if car_vel > 2.0:
            curve_target_speed = 10.0 - 6.0 * abs(curve_ind)
            speed_excess = max(0.0, car_vel - curve_target_speed)
            curve_speed_penalty = -0.1 * speed_excess    # softened

        # 2b. LATERAL BALANCE REWARD - always positive, highest when centred
        balance_reward = 0.4 * (1.0 - min(1.0, abs(curve_ind)))  # unchanged

        # 2c. Continuous safety penalty (boosted base)
        car_half_width = 1.0
        lane_half_width = 6.0
        gap = max(min_lidar - car_half_width, 0.0)
        s = np.clip(gap / (lane_half_width - car_half_width), 0.0, 1.0)
        # Steer-away bonus: small positive if steering away from nearer wall when close
        wall_side = np.sign(curve_ind)  # +1 if right wall closer (steer left, negative), -1 if left (steer right, positive)
        is_steering_away = np.sign(steer) == -wall_side  # opposite sign to wall_side
        steer_bonus = 0.1 * (1.0 - s) if is_steering_away and gap < 4.0 else 0.0

        safety_penalty = -1.5 * (1.0 - s) ** 2      # stronger negative pull

        # Combine and clip final reward (keep total in [-1,1])
        reward = progress_reward + speed_reward + curve_speed_penalty + balance_reward + steer_bonus + safety_penalty
        reward = float(np.clip(reward, -1.0, 1.0))
        # Store breakdown for debugging
        self._last_reward_breakdown = {
            'progress': float(progress_reward),
            'speed': float(speed_reward),
            'curve_penalty': float(curve_speed_penalty),
            'balance': float(balance_reward),
            'steer_bonus': float(steer_bonus),
            'safety': float(safety_penalty)
        }
        
        return reward
    
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
