# Simple 3D RL Car Project

## Setup
1. Create and activate virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Training
Run:
```
python train.py
```
Trains PPO for 200k timesteps, saves to `models/ppo_car.zip`.

## Evaluation
Run:
```
python eval.py
```
Loads model, runs episode, saves video to `videos/eval.mp4`. For interactive 3D view, set render_mode='human' in eval.py.

## Notes
- Environment: Simple oval track with basic car physics.
- Goal: Complete 3 laps.
- Visualization: Uses MuJoCo renderer; video saved as MP4.
