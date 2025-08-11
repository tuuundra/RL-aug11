#!/usr/bin/env python3
"""
Debug script to track car's journey in detail
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.simple_car_env import SimpleCarEnv
import math
import os
import glob
import re

def debug_car_journey(model_path: str = None, max_steps: int = 2000):
    """Track and visualize the car's complete journey"""
    
    # Load latest model if not specified
    if model_path is None:
        run_dirs = sorted(glob.glob("runs/run_*"), key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))
        if not run_dirs:
            raise ValueError("No run folders found in runs/")
        latest_run = run_dirs[-1]
        run_num = int(re.search(r'run_(\d+)', latest_run).group(1))
        model_path = f"{latest_run}/ppo_car.zip"
    else:
        # Extract run_num from provided path if possible
        match = re.search(r'run_(\d+)/ppo_car\.zip', model_path)
        run_num = int(match.group(1)) if match else 0
    
    print(f"Debugging model: {model_path} (run {run_num})")
    
    # Load model and environment
    env = SimpleCarEnv(render_mode=None)
    model = PPO.load(model_path)
    
    # Track journey data
    positions = []
    angles = []
    actions = []
    rewards = []
    lap_progress = []
    
    obs, _ = env.reset()
    print("=== DETAILED CAR JOURNEY TRACKING ===")
    print(f"Track dimensions: {env.track_a}×{env.track_b}m oval")
    print(f"Starting position: ({obs[0]:.2f}, {obs[1]:.2f})")
    print()
    
    for step in range(max_steps):
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, _, info = env.step(action)
        
        # Record data
        pos = obs[0:2]  # x,y (ignore z for 2D plot)
        angle = obs[6]
        positions.append(pos.copy())
        angles.append(angle)
        actions.append(action.copy())
        rewards.append(reward)
        lap_progress.append(info.get('laps', 0))
        
        # Print every 10 steps
        if step % 10 == 0 or step < 10:
            print(f"Step {step:3d}: pos=({pos[0]:6.2f}, {pos[1]:6.2f}), angle={angle:6.3f}")
            print(f"          action=[{action[0]:5.2f}, {action[1]:5.2f}], reward={reward:6.2f}")
            print(f"          laps={info.get('laps', 0)}")
            print()
        
        if done:
            print(f"Episode ended at step {step}")
            print(f"Final reward: {sum(rewards):.2f}")
            print(f"Laps completed: {lap_progress[-1]}")
            break
    
    # Convert to numpy arrays
    positions = np.array(positions)
    actions = np.array(actions)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Car trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=2, label='Car path')
    ax1.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
    
    # Plot track boundaries (oval)
    theta = np.linspace(0, 2*np.pi, 100)
    outer_x = (env.track_a + 0.5) * np.cos(theta)  # rough outer
    outer_y = (env.track_b + 0.5) * np.sin(theta)
    ax1.plot(outer_x, outer_y, 'k--', alpha=0.5, label='Track outer')
    inner_x = (env.track_a - 0.5) * np.cos(theta)
    inner_y = (env.track_b - 0.5) * np.sin(theta)
    ax1.plot(inner_x, inner_y, 'k--', alpha=0.5, label='Track inner')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Car Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Actions over time
    steps = range(len(actions))
    ax2.plot(steps, actions[:, 0], label='Throttle', linewidth=2)
    ax2.plot(steps, actions[:, 1], label='Steering', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Actions Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rewards over time
    ax3.plot(steps, rewards, linewidth=2, color='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Reward')
    ax3.set_title('Rewards Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Lap progress
    ax4.plot(steps, lap_progress, linewidth=2, color='orange')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Current Lap')
    ax4.set_title('Lap Progress')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save PNG in run folder with sub-numbering
    latest_run = os.path.dirname(model_path)
    sub_num = 1
    png_path = f'{latest_run}/car_journey_debug_{sub_num}.png'
    while os.path.exists(png_path):
        sub_num += 1
        png_path = f'{latest_run}/car_journey_debug_{sub_num}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{png_path}'")
    
    # Summary statistics
    print("\n=== JOURNEY SUMMARY ===")
    print(f"Total steps: {len(positions)}")
    print(f"Distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f}m")
    print(f"Max X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
    print(f"Max Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
    print(f"Average throttle: {actions[:, 0].mean():.3f}")
    print(f"Average steering: {actions[:, 1].mean():.3f}")
    print(f"Final laps: {lap_progress[-1]}")
    
    env.close()
    return positions, actions, rewards

if __name__ == "__main__":
    debug_car_journey()  # Loads latest model by default
