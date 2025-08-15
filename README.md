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

## 4  Current reward design (v3)

Track lane width = **12 m**, car width = **2 m** (1 m half-width). Lidar beams are normalised by a 40 m cap; in-lane readings therefore fall in roughly 0 – 0.85.

| Component | Formula (per-step) | Purpose | Scale |
|-----------|--------------------|---------|-------|
| **Progress** | `tanh( forward_speed_along_centerline / 10 )` | Primary objective – move clockwise | ≈ −1 … +1 |
| **Speed / idle** | `−0.2` if `v < 1 m/s` else `1.5 · tanh(v / 2)` | Encourages ≥1 m/s, caps at ~6 m/s | −0.2 … +1.5 |
| **Curve-speed penalty** | `−0.3 · max(0, v − (10 − 6·|curve_ind|))` | Slows before tight bends | 0 … −∞ (clipped later) |
| **Balance reward** | `0.4 · (1 − |curve_ind|)` | Rewards staying centred; still allows inside-line racing | 0 … +0.4 |
| **Safety penalty** | `0` if `min_lidar ≥ 3 m` else `−2·(3 − min_lidar)` | Punishes drifting <3 m from either wall | 0 … −6 |
| **Off-track** | `−100` and terminate | Hard crash signal | −100 |

After summation the reward is **clipped to [−1, +1]** before being returned to PPO.  Thus the relative magnitudes guide learning, while the clip keeps the value head stable.

`curve_ind` is computed from lidar symmetry:
```
curve_ind = − [(L60 + L30) − (R30 + R60)] / 2    # ∈ [−1, +1] roughly
```
Positive means the right wall is closer (clockwise turn), negative means the left wall.

> Recent changes
> • Lidar max-range reduced from 100 m → 40 m for better resolution.  
> • Added continuous **balance_reward** (v3).  
> • Safety penalty and curve-speed parameters to be tuned next.

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
