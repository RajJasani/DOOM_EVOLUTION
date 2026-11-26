# Member 4 Handoff: Reinforcement Learning & Training Pipeline

**Target Role:** Member 4 (Reinforcement Learning & Training)  
**Lead Coordinator:** Member 1 (AI Architecture & Integration)  
**Target Module:** `training/`  
**Reference Specs:** `docs/CONTRACT.md`, `INTERFACES.md`, `docs/EVALUATION.md`  

---

## 1. Integrated Architecture Overview & Pre-Flight Status

Following the Phase 1–6 Integration Repairs, the perception, recurrent neural network, and agent submission modules are fully unified.

Key integration characteristics:
- **Observation Pipeline Parity:** Identical preprocessing via `ObservationAdapter` (inference) and `VectorObservationAdapter` (training).
- **Canonical Vitals Normalization:** Normalization occurs strictly once inside the perception pipeline. `NetworkConfig.normalize_vitals` is set to `False` by default so `DoomNetwork` does not double-normalize.
- **Spatial Memory Integration:** Member 2's `memory_features` (64-dim egocentric spatial memory) are connected to Member 3's `DoomNetwork` via `MemoryEncoder` (64 $\to$ 64 MLP) and fused in `FeatureFusion` (320 $\to$ 256).
- **Depth Processing Configurable:** `enable_depth=False` by default in adapters and agent inference, saving significant CPU overhead during high-throughput rollouts.
- **Reward Shaper Verified:** `DoomRewardShaper` supports both positional and keyword invocations (`env_reward` or `raw_env_reward`) and correctly handles health damage penalties in true HP units across raw or normalized vitals.

---

## 2. Observation Pipeline for Parallel Training Rollouts

Do not pass raw ViZDoom observation dictionaries directly to `DoomNetwork`. Use the canonical `VectorObservationAdapter`.

### 2.1 Multi-Environment Rollout Setup

```python
from perception import VectorObservationAdapter
from network import DoomNetwork, NetworkConfig

num_envs = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize vectorized adapter (depth disabled by default to save CPU)
adapter = VectorObservationAdapter(num_envs=num_envs, enable_depth=False)
model = DoomNetwork(NetworkConfig()).to(device)

# 2. Reset environments and adapter slots
raw_obs_list = [envs[i].reset()[0] for i in range(num_envs)]
for i in range(num_envs):
    adapter.reset_env(i)

# 3. Process observations through adapter
processed_obs_list = adapter.process_batch(raw_obs_list)

# 4. Collate into batched PyTorch tensors for DoomNetwork
batched_tensors = adapter.collate_for_network(processed_obs_list, device=device)
```

### 2.2 Per-Environment Episode Termination

When environment $i$ terminates or truncates:
1. Reset the environment: `raw_obs_list[i] = envs[i].reset()[0]`
2. Reset that specific environment's spatial memory slot: `adapter.reset_env(i)`
3. Reset that environment's recurrent hidden state: `hx[:, i, :] = 0.0`

---

## 3. Observation Dictionary Schema & Tensor Shapes

When collated via `adapter.collate_for_network()`, the observation dictionary has the following layout:

| Key | Nested Key | Type / Dtype | Batch Shape `(B, ...)` | Sequential Shape `(B, T, ...)` | Semantics & Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vitals` | — | `torch.float32` | `(B, 17)` | `(B, T, 17)` | Pre-normalized: Health `[0, 1]`, Armor `[0, 1]`, Weapons `[0, 1]`, Ammo `[0, 1]`, Velocity `[-1, 1]`. |
| `rays` | `distance` | `torch.float32` | `(B, 64)` | `(B, T, 64)` | Normalized distance to nearest hit in `[0.0, 1.0]`. |
| `rays` | `hit_type` | `torch.int64` | `(B, 64)` | `(B, T, 64)` | Surface type enum: 0=None, 1=Wall, 2=Door, 3=Enemy, 4=Pickup, 5=Obstacle. |
| `visible_entities` | `features` | `torch.float32` | `(B, 32, 5)` | `(B, T, 32, 5)` | Normalized `[dist, sin(yaw), cos(yaw), pitch, relative_vel]`. |
| `visible_entities` | `entity_type` | `torch.int64` | `(B, 32)` | `(B, T, 32)` | Entity category enum: 0..8 (Player, Shotgunner, Imp, etc.). |
| `visible_entities` | `mask` | `torch.float32` | `(B, 32)` | `(B, T, 32)` | 1.0 for valid detected entity slot, 0.0 for padding. |
| `memory_features` | — | `torch.float32` | `(B, 64)` | `(B, T, 64)` | Egocentric decay-weighted spatial memory vector (doors, enemies, pickups, breadcrumbs). |

---

## 4. Recurrent PPO Training Loop (`training/`)

### 4.1 Rollout Step (`training/rollout.py`)

```python
with torch.no_grad():
    step_res = model.step(batched_tensors, hx=hx, deterministic=False)

actions_np = step_res.action.cpu().numpy().astype(np.int8)  # Shape (B, 9)
values = step_res.value  # Shape (B, 1)
log_probs = step_res.log_prob  # Shape (B,)
next_hx = step_res.hx  # Shape (1, B, 256)

# Step vectorized environments
for i in range(num_envs):
    next_raw_obs, env_reward, term, trunc, info = envs[i].step(actions_np[i])
    done = term or trunc
    
    shaped_reward = reward_shaper[i].compute_reward(
        obs=processed_obs_list[i],
        env_reward=env_reward,
        terminated=term,
        truncated=trunc,
    )
    
    # Store transition in buffer:
    # obs[i], actions[i], shaped_reward, values[i], log_probs[i], done, hx[:, i:i+1, :]
    
    if done:
        next_raw_obs = envs[i].reset()[0]
        adapter.reset_env(i)
        reward_shaper[i].reset(processed_obs_list[i])
        next_hx[:, i, :] = 0.0
        
    raw_obs_list[i] = next_raw_obs
    
hx = next_hx
```

### 4.2 Generalized Advantage Estimation (GAE)
Compute advantages across the rollout horizon $T$:
$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$A_t = \delta_t + (\gamma \lambda) (1 - d_t) A_{t+1}$$
$$\hat{R}_t = A_t + V(s_t)$$
Recommended hyperparameters: $\gamma = 0.99$, $\lambda = 0.95$. Normalize advantages across the mini-batch:
$$A_{\text{norm}} = \frac{A - \mu_A}{\sigma_A + 10^{-8}}$$

### 4.3 Recurrent Mini-Batch Loss (`training/ppo.py`)
Chunk rollout trajectories into sequence segments of length $L$ (e.g., $L = 16$ or $32$ steps) to perform truncated Backpropagation Through Time (BPTT):

```python
# seq_obs is a dict of tensors with shapes (B_sub, L, ...)
# actions_chunk shape (B_sub, L, 9)
# dones_chunk shape (B_sub, L) -> used by model to reset GRU state mid-sequence
eval_res = model.evaluate_actions(
    obs=seq_obs,
    hx=hx_start,  # Shape (1, B_sub, 256)
    actions=actions_chunk,
    dones=dones_chunk,
)

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

## 5. Action Space & Edge Triggering

The action space is strictly `MultiBinary(9)`:
`[ATTACK, SPEED, STRAFE, MOVE_R, MOVE_L, TURN_R, TURN_L, USE, SELECT_NEXT_WEAPON]`

- **Policy Distribution:** Independent Bernoulli distributions per button.
- **Continuous Button Presses:** In ViZDoom, buttons like `USE` (index 7) or `SELECT_NEXT_WEAPON` (index 8) can stall if held down continuously across multiple consecutive tics.
- **ActionEdgeTriggerAdapter:** Located in `perception.action_adapter.ActionEdgeTriggerAdapter`.
  - In `submission/agent.py`, it is instantiated with `enabled=False` by default to preserve the exact raw network output.
  - If your trained policy exhibits weapon toggling or door stuckness due to holding buttons, you can safely enable `ActionEdgeTriggerAdapter(enabled=True)` in inference or during evaluation.

---

## 6. Training Curriculum & Milestone Targets

Progress through map curricula:

| Stage | Objective | Primary Reward Drivers | Map Types |
| :---: | :--- | :--- | :--- |
| **Stage 1: Combat & Survival** | Aiming, shooting, dodging projectiles, weapon management. | Kill bonus (+1.0), damage penalty avoidance (-0.05/HP), death penalty (-2.0). | Small arenas with Shotgun Guy / Imp spawns. |
| **Stage 2: Multi-Room Navigation** | Hallway traversal, clearing dead ends, opening doors. | Forward exploration bonus (+0.02), door opening incentive. | `two_rooms.wad` and multi-room layouts. |
| **Stage 3: Progression & Keys** | Picking up keycards, remembering locked door locations, backtracking. | Keycard pickup (+3.0), locked door resolution. | Locked-door WADs requiring Blue/Yellow/Red keys. |
| **Stage 4: Exit Completion** | Navigating gauntlet from start to exit switch. | Exit milestone completion (+10.0). | Full multi-stage levels. |

---

## 7. Checkpoint Delivery Requirements

1. **Target File Path:** `submission/weights/model.pt`
2. **Contents:** `model.state_dict()` (strictly matching `DoomNetwork` keys and tensor shapes).
3. **Format & Tracking:**
   - Saved with standard `torch.save(model.state_dict(), "submission/weights/model.pt")`.
   - Checkpoint size: ~4.6 MB (well under the 1 GB repository submission limit).
4. **Verification Step:**
   - Run `python validate.py` from repository root.
   - Run `python examples/forward_pass_example.py` to confirm inference.
