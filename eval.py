import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_car_env import SimpleCarEnv
import imageio
import numpy as np
import os
import glob
import re

def main():
    env = SimpleCarEnv(render_mode='rgb_array')
    # Find latest run folder
    run_dirs = sorted(glob.glob("runs/run_*"), key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))
    if not run_dirs:
        raise ValueError("No run folders found in runs/")
    latest_run = run_dirs[-1]
    run_num = int(re.search(r'run_(\d+)', latest_run).group(1))
    latest_model = f"{latest_run}/ppo_car.zip"
    model = PPO.load(latest_model)
    
    obs, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        frame = env.render()
        frames.append(frame)
        
        if done:
            print(f"Completed {info['laps']} laps")
    
    # Save video in run folder with sub-numbering
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
