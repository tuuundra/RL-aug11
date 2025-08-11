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
    # Find latest model
    model_files = sorted(glob.glob("models/ppo_car_*.zip"))
    if not model_files:
        raise ValueError("No models found in models/")
    latest_model = model_files[-1]
    model_num = int(re.search(r'ppo_car_(\d+)\.zip', latest_model).group(1))
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
    
    # Save video with numbering
    os.makedirs('videos', exist_ok=True)
    sub_num = 1
    video_path = f'videos/eval_{model_num}.mp4'
    while os.path.exists(video_path):
        sub_num += 1
        video_path = f'videos/eval_{model_num}_{sub_num}.mp4'
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")
    
    env.close()
    print("Evaluation complete! Check videos/eval.mp4")

if __name__ == "__main__":
    main()
