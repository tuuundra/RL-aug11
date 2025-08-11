import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.simple_car_env import SimpleCarEnv
import os

def main():
    # Create vectorized environment
    env = make_vec_env(SimpleCarEnv, n_envs=4)
    
    # PPO with basic hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1
    )
    
    # Train
    model.learn(total_timesteps=200_000)
    
    # Save with incremental numbering
    os.makedirs('models', exist_ok=True)
    model_num = 1
    while os.path.exists(f"models/ppo_car_{model_num}.zip"):
        model_num += 1
    model_path = f"models/ppo_car_{model_num}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
