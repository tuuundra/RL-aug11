# Oval-Track RL Car (MuJoCo + PPO)

## 1  Quick start
1. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2  Training
Training is handled by **`train_puffer.py`**, which automatically creates numbered run folders:
```bash
python train_puffer.py            # default 2 M steps
```
Key flags:
* `--timesteps N`   override total steps (e.g. `300000` for quick tests)
* `--continue-from runs/run_XX/ppo_car.zip`   resume / fine-tune an existing model

## 3  Evaluation & debugging
Run:
```bash
python debug_car_journey.py runs/run_23/ppo_car.zip   # 1 episode, detailed log + PNG
python eval_puffer.py        runs/run_23/ppo_car.zip   # multi-episode, saves MP4
```

## 4  Reward scheme history
| Version | Observation / Reward features | Representative run | Result |
|---------|--------------------------------|-------------------|---------|
| v0 | Moving waypoints, flat speed bonus | pre-fix baseline | crashes < 200 steps |
| v1 | Fixed waypoints, balanced terms, privileged `next_curve_angle` | run 20 | ~1900 steps, no lap |
| v2 | Lidar-based curve indicator, curve-speed penalty, braking allowed | runs 21-23 | agent creeps, still no lap |

## 5  Next directions
* Increase positive speed shaping to break “idle” local optimum.
* Curriculum: ramp penalties only after reaching 2 m/s.
* Analyse reward breakdowns to tune safety vs. progress terms.

## 6  Repository layout
```
envs/                      # MuJoCo environment definition
train_puffer.py            # main training script
debug_car_journey.py       # detailed single-episode tracker
eval_puffer.py             # batch evaluation & video
runs/                      # auto-generated experiment folders
```
