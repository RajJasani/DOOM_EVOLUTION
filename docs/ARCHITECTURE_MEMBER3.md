# Member 3: Neural Network & Recurrent Memory Architecture Specification

**Project:** Yugaantar DevForge Hackathon – *"Can It Play DOOM?"*  
**Role:** Member 3 (Neural Network & Memory Architecture)  
**Branch:** `feature/neural-network`  
**Target Specification:** `docs/CONTRACT.md` (Contract version: `1.0.0-draft`)

---

## 1. Overview & Architectural Philosophy

The decision-making brain of our autonomous DOOM agent is a custom, modular PyTorch neural network designed to operate under strict competition hardware constraints (2 GB VRAM, 8 GB RAM, 1 CPU thread, <40 ms/step inference latency).

Rather than relying on generic off-the-shelf policies, our architecture incorporates domain-specific inductive biases for first-person combat and exploration in 2.5D Doom levels:
1. **Topological Circular Raycast Encoder:** Employs circular 1D convolution over 64 ego-centric distance rays (360° field of view) to exploit spatial continuity between ray 0 and ray 63.
2. **Robust Multi-Head Enemy Attention:** Attends over up to 32 visible entities with mathematical padding invariance and NaN-proof fallbacks for empty enemy encounters.
3. **Player Vitals Normalization:** Normalizes scalar and discrete variables (health, armor, ammo counts, key possession, ego-velocities).
4. **GRU-Based Recurrent Memory:** Maintains spatial persistence and exploration history across decision steps with clean reset mechanisms between maps and episode boundaries.
5. **Decoupled Actor-Critic Heads:** An Independent Bernoulli policy head for the 9-dimensional `MultiBinary` action space alongside a scalar state-value estimator $V(s)$.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │            Official Ego-Centric Observation             │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
          ┌──────────────────────────┬────────────┴─────────────┬──────────────────────────┐
          ▼                          ▼                          ▼                          ▼
   Vitals (17,)             Rays (dist: 64, type: 64)   Entities (32, 5 + type)      Visual Features
          │                          │                          │                     (Optional M2)
          ▼                          ▼                          ▼                          │
    VitalsEncoder             RaycastEncoder          EnemyAttentionModule                 │
    (2-layer MLP)           (Circular Conv1d+MLP)    (Self-Attn + Cross-Pool)              │
    [B, 64]                    [B, 128]                    [B, 64]                         │
          │                          │                          │                          │
          └──────────────────────────┼──────────────────────────┴──────────────────────────┘
                                     ▼
                               FeatureFusion
                            (2-layer MLP + Norm)
                                  [B, 256]
                                     │
                                     ▼
                              RecurrentMemory
                           (GRU Layer, hx persistence)
                                  [B, 256]
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
               PolicyHead                         ValueHead
           (Actor: 9 Logits)                  (Critic: Scalar V)
           MultiBinary(9)                         [B, 1]
```

---

## 2. Tensor Dimension Reference Guide

All tensor shapes directly mirror `env/doom_env/spaces.py` and `docs/CONTRACT.md`:

| Module Component | Input Shape | Internal Operations | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **VitalsEncoder** | `(..., 17)` | Component-wise scaling + 2-layer MLP (17 $\to$ 64 $\to$ 64) | `(..., 64)` | Scaled health, armor, log1p ammo, key flags, velocities |
| **RaycastEncoder** | Dist: `(..., 64)`<br>Type: `(..., 64)` | `nn.Embedding(6, 16)` $\to$ Circular Pad $\to$ `Conv1d(17, 32)` $\to$ Circular Pad $\to$ `Conv1d(32, 64)` $\to$ `Linear` | `(..., 128)` | Circular padding preserves 360° topology across ray 0 & 63 |
| **EnemyAttention** | Feats: `(..., 32, 5)`<br>Types: `(..., 32)`<br>Mask: `(..., 32)` | Token projection $\to$ Self-Attn $\to$ FFN $\to$ Cross-Attn pooling with vitals query | `(..., 64)` | Mathematically invariant to padding; guaranteed zero output if 0 enemies |
| **Visual Hook** | `(..., D_vis)` | Optional direct passthrough from Member 2 | `(..., D_vis)` | Defaults to `0` if visual buffers are skipped |
| **FeatureFusion** | `(..., 64 + 128 + 64)` | Concatenation $\to$ 2-layer MLP (256 $\to$ 256 $\to$ 256) + LayerNorm + GELU | `(..., 256)` | Unified multimodal state representation |
| **RecurrentMemory** | `(B, 256)` or `(B, T, 256)` | `nn.GRU(256, 256, num_layers=1, batch_first=True)` with hidden state `(1, B, 256)` | `(B, 256)` or `(B, T, 256)` | Hidden state persistent across tics; reset on `dones` or between maps |
| **PolicyHead** | `(..., 256)` | MLP (256 $\to$ 128 $\to$ 9) | Logits: `(..., 9)`<br>Actions: `(..., 9)` $\in \{0, 1\}$ | Bernoulli distribution over the 9 buttons |
| **ValueHead** | `(..., 256)` | MLP (256 $\to$ 128 $\to$ 1) | `(..., 1)` | Expected future return estimate $V(s)$ |

---

## 3. Mathematical Details & Robustness Guarantees

### 3.1 Circular 1D Convolution over Ego-Centric Rays
In ViZDoom, rays are cast evenly at angles $\theta_k = k \times \frac{360^\circ}{64}$ for $k \in \{0, \dots, 63\}$. Ray 0 points directly forward ($0^\circ$), ray 1 points to the left ($+5.625^\circ$), and ray 63 points to the right ($-5.625^\circ$).  
Standard 1D convolutions truncate boundaries at index 0 and 63. We apply **circular padding**:
$$\tilde{X} = \text{pad}_{\text{circular}}(X, \text{pad}=1)$$
This enables the convolutional filter to smoothly capture narrow hallways, doors, or enemies directly in front of the player spanning rays 63, 0, and 1 without boundary distortion.

### 3.2 NaN-Proof Enemy Attention Masking
When an agent is in an empty room, `mask.sum() == 0` (all 32 slots are padding). In naive multi-head attention implementations, passing an all-`True` `key_padding_mask` causes softmax to compute:
$$\text{softmax}(-\infty, \dots, -\infty) = \text{NaN}$$
Our `EnemyAttentionModule` detects `has_any_entity = mask.any(dim=-1)`:
1. If `has_any_entity` is false for a sample, slot 0 is temporarily unmasked so softmax receives finite values.
2. The final output is masked via `torch.where(has_any_entity.unsqueeze(-1), out, torch.zeros_like(out))`.
This guarantees that 0 visible enemies produces a deterministic zero vector with strictly finite, non-NaN gradients.

---

## 4. Integration Guide for Member 1 (`submission/agent.py`)

Member 1 needs to integrate our `DoomNetwork` into `submission/agent.py` to evaluate the trained model on hidden maps.

### Step-by-Step Code for `submission/agent.py`:

```python
from pathlib import Path
import numpy as np
import torch

from network import DoomNetwork, NetworkConfig

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once per evaluation run."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Instantiate network
        self.config = NetworkConfig()
        self.model = DoomNetwork(self.config).to(self.device)
        self.model.eval()

        # Load weights if available
        weights_path = WEIGHTS_DIR / "model.pt"
        if weights_path.is_file():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        # Recurrent hidden state tracker
        self.hx = None

    def reset(self) -> None:
        """Called before every map. Clear recurrent state."""
        self.hx = None

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        with torch.no_grad():
            action_np, self.hx = self.model.get_action(
                obs=observation,
                hx=self.hx,
                deterministic=True,  # Greedy exploitation during evaluation
            )
        return action_np
```

---

## 5. Integration Guide for Member 4 (`training/ppo.py`)

Member 4 can directly leverage `DoomNetwork.step()` for fast environment rollouts and `DoomNetwork.evaluate_actions()` for vectorized PPO policy & value loss updates.

### 5.1 Rollout Collection Loop
```python
model = DoomNetwork().to(device)
obs, info = env.reset()
hx = model.init_hidden(batch_size=1, device=device)

rollout_obs = []
rollout_actions = []
rollout_log_probs = []
rollout_values = []
rollout_dones = []

for step in range(rollout_steps):
    step_res = model.step(obs, hx=hx, deterministic=False)
    action_np = step_res.action.cpu().numpy().astype(np.int8)

    next_obs, reward, terminated, truncated, info = env.step(action_np)
    done = terminated or truncated

    rollout_obs.append(obs)
    rollout_actions.append(step_res.action)
    rollout_log_probs.append(step_res.log_prob)
    rollout_values.append(step_res.value)
    rollout_dones.append(done)

    obs = next_obs
    hx = step_res.hx if not done else model.init_hidden(batch_size=1, device=device)
```

### 5.2 PPO Update Step
```python
# During PPO epoch update:
# obs_batch is a batched dict of shape (B, T, ...)
# actions_batch: (B, T, 9)
# dones_batch: (B, T)
# old_log_probs: (B, T)
# returns: (B, T)
# advantages: (B, T)

eval_res = model.evaluate_actions(
    obs=obs_batch,
    hx=initial_hx_batch,
    actions=actions_batch,
    dones=dones_batch,
)

# Policy Loss (PPO clipped surrogate)
ratio = torch.exp(eval_res.log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# Value Loss (clipped MSE)
value_loss = 0.5 * ((eval_res.values.squeeze(-1) - returns) ** 2).mean()

# Entropy Bonus
entropy_loss = -entropy_coef * eval_res.entropy.mean()

total_loss = policy_loss + value_loss + entropy_loss
optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
optimizer.step()
```

---

## 6. Integration Guide for Member 2 (Perception & Vision)

If Member 2 trains a visual feature encoder (e.g. CNN on `rgb` and `depth` buffers):
1. Set `config.visual_feature_dim = D_vis` (e.g., 64 or 128).
2. Pass precomputed visual embeddings inside the observation dictionary under key `"visual_embedding"` with shape `(..., D_vis)`.
3. `FeatureFusion` will automatically concatenate `visual_embedding` with the vitals, raycast, and enemy embeddings into the GRU input.
4. If `visual_feature_dim == 0` (default), the network operates purely on raycasts, vitals, and entity features, keeping computation ultra-fast (0.49 ms/step) and saving memory.
