# Member 4 Handoff: Reinforcement Learning & Training Pipeline

**Target Role:** Member 4 (Reinforcement Learning & Training)  
**Lead Coordinator:** Member 1 (AI Architecture & Integration)  
**Target Module:** `training/`  
**Reference Specs:** `docs/CONTRACT.md`, `INTERFACES.md`, `docs/EVALUATION.md`  

---

## 1. Current Implementation Status

- **Environment Wrapper Verified:** `doom_env.make_env(wad_path, map_name, episode_ticks=2100)` is verified and operational.
- **Reward Shaping Foundation Created:** `training/rewards.py` implements `DoomRewardShaper`:
  - Milestone bonus (+10.0 for reaching exit).
  - Combat survival (+1.0 per enemy kill, -2.0 on death).
  - Keycard progression (+3.0 per keycard picked up).
  - Health preservation (-0.05 per damage point taken).
  - Anti-camping / exploration incentive (+0.02 when moving).
- **Network Interface Ready:** `network.DoomNetwork` exposes `step()` for rollout collection and `evaluate_actions()` for PPO loss computation.

---

## 2. Core Technical Requirements for Member 4

### 2.1 Rollout Collection Loop (`training/rollout.py`)
Collect transitions across $N$ parallel environments or sequential episodes:
- **Per Step:**
  ```python
  step_res = model.step(obs, hx=hx, deterministic=False)
  action_np = step_res.action.cpu().numpy().astype(np.int8)
  next_obs, env_reward, terminated, truncated, info = env.step(action_np)
  done = terminated or truncated

  shaped_reward = reward_shaper.compute_reward(next_obs, env_reward, terminated, truncated)
  
  # Store in rollout buffer:
  # obs, action, shaped_reward, value, log_prob, done, hx
  
  obs = next_obs
  hx = step_res.hx if not done else model.init_hidden(batch_size=1, device=device)
  ```

### 2.2 Generalized Advantage Estimation (GAE)
Compute advantages across the rollout horizon $T$:
$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$A_t = \delta_t + (\gamma \lambda) (1 - d_t) A_{t+1}$$
$$\hat{R}_t = A_t + V(s_t)$$
Use $\gamma = 0.99$ and $\lambda = 0.95$. Normalize advantages across the mini-batch:
$$A_{\text{norm}} = \frac{A - \mu_A}{\sigma_A + 10^{-8}}$$

### 2.3 Recurrent PPO Loss Calculation (`training/ppo.py`)
Chunk rollout trajectories into sequence segments of length $L$ (e.g., $L = 16$ or $32$ steps) to perform truncated Backpropagation Through Time (BPTT):
```python
eval_res = model.evaluate_actions(obs_chunk, hx_start, actions_chunk, dones_chunk)

# 1. Clipped Surrogate Policy Loss
ratio = torch.exp(eval_res.log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# 2. Value Function Loss (Clipped MSE)
val_clipped = old_values + torch.clamp(eval_res.values - old_values, -clip_eps, clip_eps)
val_loss_unclipped = (eval_res.values.squeeze(-1) - returns) ** 2
val_loss_clipped = (val_clipped.squeeze(-1) - returns) ** 2
value_loss = 0.5 * torch.max(val_loss_unclipped, val_loss_clipped).mean()

# 3. Entropy Regularization
entropy_loss = -entropy_coef * eval_res.entropy.mean()

total_loss = policy_loss + 0.5 * value_loss + entropy_loss

optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
optimizer.step()
```

---

## 3. Training Curriculum & Milestone Targets

Design `training/train.py` to progress through the competition map categories:

| Stage | Objective | Primary Reward Drivers | Map Types |
| :---: | :--- | :--- | :--- |
| **Stage 1: Combat & Survival** | Aiming, shooting, dodging projectiles, weapon management. | Kill bonus, damage penalty avoidance, health preservation. | Small arenas with Shotgun Guy / Imp spawns. |
| **Stage 2: Multi-Room Navigation** | Hallway traversal, clearing dead ends, opening doors. | Forward exploration bonus, door opening incentive. | `two_rooms.wad` and multi-room layouts. |
| **Stage 3: Progression & Keys** | Picking up keycards, remembering locked door locations, backtracking. | Keycard pickup (+3.0), locked door resolution. | Locked-door WADs requiring Blue/Yellow/Red keys. |
| **Stage 4: Exit Completion** | Navigating gauntlet from start to exit switch. | Exit milestone completion (+10.0). | Full multi-stage levels. |

---

## 4. Checkpoint Delivery Requirements

1. **Target File Path:** `submission/weights/model.pt`
2. **Contents:** `model.state_dict()` (strictly matching `DoomNetwork` keys and tensor shapes).
3. **Format & Tracking:**
   - Saved with standard `torch.save(model.state_dict(), "submission/weights/model.pt")`.
   - Tracked via **Git LFS** (`git lfs track "*.pt"`).
   - Checkpoint size must remain **< 10 MB** (limit for submission is 1 GB total).
4. **Validation Test:**
   - After saving, run `python validate.py` to ensure the agent loads the weights and successfully executes full episodes without crashes.
