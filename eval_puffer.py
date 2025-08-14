import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_car_env import SimpleCarEnv
import imageio
import numpy as np
import os
import glob
import re

def main():
    # Create environment for evaluation (single env, with rendering)
    env = SimpleCarEnv(render_mode='rgb_array', eval_mode=True)
    
    # Find latest run folder (matching eval.py conventions)
    run_dirs = sorted(glob.glob("runs/run_*"), key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))
    if not run_dirs:
        print("No run folders found in runs/. Please train first with train_puffer.py")
        return
    latest_run = run_dirs[-1]
    run_num = int(re.search(r'run_(\d+)', latest_run).group(1))
    model_path = f"{latest_run}/ppo_car.zip"
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please train first.")
        return
    
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    done = False
    frames = []
    step = 0
    
    # Initial telemetry
    pos = env.data.qpos[0:2]
    vel = np.linalg.norm(env.data.qvel[0:2])
    min_lidar = min(obs[5:10]) * env.max_lidar_dist
    print(f"step={step} pos=[{pos[0]:.2f}, {pos[1]:.2f}] vel={vel:.2f} m/s min_lidar={min_lidar:.2f} waypoint={env.current_waypoint} laps={env.current_lap} on_track={env._is_on_track(pos[0], pos[1])}")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        step += 1
        
        # Per-step telemetry
        pos = env.data.qpos[0:2]
        vel = np.linalg.norm(env.data.qvel[0:2])
        min_lidar = min(obs[5:10]) * env.max_lidar_dist
        on_track = env._is_on_track(pos[0], pos[1])
        print(f"step={step} pos=[{pos[0]:.2f}, {pos[1]:.2f}] vel={vel:.2f} m/s min_lidar={min_lidar:.2f} waypoint={env.current_waypoint} laps={env.current_lap} on_track={on_track}")
        
        frame = env.render()
        frames.append(frame)
        
        if done:
            print(f"Episode terminated after {step} steps. Final laps: {info['laps']}")
    
    # Save video in run folder with sub-numbering (matching eval.py conventions)
    sub_num = 1
    video_path = f'{latest_run}/eval_{sub_num}.mp4'
    while os.path.exists(video_path):
        sub_num += 1
        video_path = f'{latest_run}/eval_{sub_num}.mp4'
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")
    
    env.close()
    print("Evaluation complete! Check videos/eval.mp4")

if __name__ == "__main__":
    main()
