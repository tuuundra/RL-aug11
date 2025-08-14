from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.simple_car_env import SimpleCarEnv
import os
import re

# Create PufferLib environment wrapper
def make_env():
    """Create a single environment instance"""
    return SimpleCarEnv()

def main():
    # Configuration
    num_envs = 8  # Number of parallel environments
    total_timesteps = 100_000  # Quick test after steering boost
    
    print(f"Creating {num_envs} parallel environments...")
    
    # Create vectorized environment using SubprocVecEnv for parallel training
    # This will run multiple environments in parallel for much faster training
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    print("Initializing PPO with vectorized environments...")
    
    # PPO with optimized settings for parallel training
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # Rollout length per environment
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001
        # Removed tensorboard_log to avoid dependency issues
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Expected speedup: ~{num_envs}x faster than single environment")
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save in new run folder (matching original train.py conventions)
    os.makedirs('runs', exist_ok=True)
    run_num = 1
    while os.path.exists(f"runs/run_{run_num}"):
        run_num += 1
    run_dir = f"runs/run_{run_num}"
    os.makedirs(run_dir, exist_ok=True)
    model_path = f"{run_dir}/ppo_car.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()
    print("Training complete!")

if __name__ == "__main__":
    main()
