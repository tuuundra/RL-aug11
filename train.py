import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.simple_car_env import SimpleCarEnv
import os
import re

def main():
    # Create vectorized environment
    env = make_vec_env(SimpleCarEnv, n_envs=4)
    
    # PPO with adjusted hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1
    )
    
    # Train
    model.learn(total_timesteps=500_000)
    
    # Save in new run folder
    os.makedirs('runs', exist_ok=True)
    run_num = 1
    while os.path.exists(f"runs/run_{run_num}"):
        run_num += 1
    run_dir = f"runs/run_{run_num}"
    os.makedirs(run_dir, exist_ok=True)
    model_path = f"{run_dir}/ppo_car.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
