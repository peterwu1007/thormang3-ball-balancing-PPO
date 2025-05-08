# Thormang3 Ball Balancing Simulation with PPO in Isaac Gym


# Ball Balancing Simulation with PPO in Isaac Gym

This project implements a deep reinforcement learning environment in Isaac Gym for training a humanoid robot arm (THORMANG3) to balance a ball on a tray. The agent is trained using Proximal Policy Optimization (PPO) with domain randomization for robustness and sim-to-real transfer.

---

##  Project Structure

```
sim2real_rl/
├── ball_balance.py            # Isaac Gym VecTask environment definition
├── BallBalance.yaml           # Environment and simulation parameters (gym config)
├── BallBalancePPO.yaml        # PPO hyperparameter configuration (rl_games format)
├── Asset                      # Contains the robo URDF description.
```
---

## Environment Overview (`ball_balance.py`)

- **Platform:** NVIDIA Isaac Gym (Preview 4)
- **Robot:** Upper-body humanoid (THORMANG3-based) with 7 DOFs
- **Task:** Balance a 1.9 cm-radius ball on a tray mounted on the robot's hand
- **Observation Space (15-D):**
  - DOF positions and velocities
  - Ball position and linear velocity (relative to tray)
  - Tray center offset vector
- **Action Space (4-D):**
  - Joint position targets (scaled and clipped)

### Reward Design
- Distance to tray center (in rotated tray frame)
- Ball linear velocity penalty
- Action smoothness penalty
- Bonus for staying centered

### Domain Randomization (in `BallBalance.yaml`)
- DOF damping and stiffness
- Ball mass
- Gravity variation
- Additive noise in actions and observations

---

## PPO Training Setup (`BallBalancePPO.yaml`)

- **Network:** MLP [512, 256, 128], ELU activation
- **Algorithm:** Clipped PPO (via `rl_games`)
- **Key Hyperparameters:**
  - `learning_rate`: 3e-4 with adaptive scheduler
  - `entropy_coef`: 0.0
  - `e_clip`: 0.2 (clipping range)
  - `normalize_advantage`: True
  - `minibatch_size`: 32768, `mini_epochs`: 8
  - `clip_value`: True
- **Extras:**
  - GAE (λ = 0.95), γ = 0.99
  - Value loss, KL penalty, adaptive LR
  - Score checkpoint saving

---

## Environment
  - python=3.7
  - pytorch=1.8.1
  - torchvision=0.9.1
  - cudatoolkit=11.1
  - pyyaml>=5.3.1
  - scipy>=1.5.0
  - tensorboard>=2.2.1



##  How to Train

1. Install Isaac Gym Preview 4 and `rl_games` library
2. Place all files in your `IsaacGymEnvs/` environment
3. Launch training with:

```bash
python train.py --cfg BallBalancePPO.yaml
```

---

## License

Based on NVIDIA Isaac Gym SDK. Includes original components adapted for custom task.  
See Isaac Gym license for redistribution terms.

---

> Maintained by [@peterwu1007](https://github.com/peterwu1007)

Based on NVIDIA Isaac Gym SDK. Includes original components adapted for custom task.  
See Isaac Gym license for redistribution terms.

---

> Maintained by [@peterwu1007](https://github.com/peterwu1007)
