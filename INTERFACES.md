# Team Integration & Module Interfaces Specification

**Project:** Yugaantar DevForge Hackathon — Track 1: *"Can It Play DOOM?"*  
**Contract Version:** `1.0.0-draft` (`docs/CONTRACT.md`)  
**Target Hardware:** 1 NVIDIA RTX 3090 (2 GB VRAM / agent), 8 GB RAM, 1 CPU thread.  
**Latency Budget:** < 40 ms per decision step (single CPU thread).  

---

## 1. Team Roles & Module Directory Layout

| Member | Role | Primary Module Directory | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | Team Lead: Architecture & Integration | `submission/`, root (`validate.py`) | Overall system architecture, evaluation entry point (`submission/agent.py`), process lifecycle, weight loading, memory reset, competition compliance. |
| **Member 2** | Spatial Intelligence & Perception | `perception/` | Raycast parsing, entity kinematics, topological feature mapping, vitals formatting, optional visual embeddings. |
| **Member 3** | Neural Network & Recurrent Memory | `network/` | Custom PyTorch models (`DoomNetwork`), circular 1D-CNN, NaN-proof multi-head entity attention, GRU recurrent memory, actor-critic heads. |
| **Member 4** | Reinforcement Learning & Training | `training/` | Recurrent PPO rollout collection, reward shaping, Generalized Advantage Estimation (GAE), curriculum learning, checkpoint export to `submission/weights/`. |

```
Doom_Final/
├── env/                     # Official ViZDoom Gymnasium wrapper (DO NOT MODIFY)
│   └── doom_env/
├── maps/                    # Test and training maps (.wad)
├── perception/              # Member 2: Spatial & perceptual feature extraction
│   ├── __init__.py
│   ├── spatial.py           # Raycast & entity kinematic geometry
│   └── vision.py            # (Optional) Compact CNN for RGB/Depth visual buffers
├── network/                 # Member 3: Neural network & recurrent memory
│   ├── __init__.py
│   ├── config.py            # NetworkConfig dataclass
│   ├── encoder.py           # VitalsEncoder, RaycastEncoder, FeatureFusion
│   ├── enemy_attention.py   # Multi-head attention over visible threats
│   ├── memory.py            # RecurrentMemory (GRU)
│   ├── model.py             # DoomNetwork unified model & StepResult
│   └── policy.py            # PolicyHead (Bernoulli 9) & ValueHead
├── training/                # Member 4: PPO, rewards, curriculum, training loop
│   ├── __init__.py
│   ├── ppo.py               # Recurrent PPO implementation
│   ├── rewards.py           # Multi-tier reward shaping functions
│   ├── rollout.py           # Vectorized rollout buffer with hidden states
│   └── train.py             # Main training entry point
├── evaluation/              # Local benchmarking & diagnostic evaluators
│   ├── __init__.py
│   └── benchmark.py         # Multi-map score & latency benchmark
├── tests/                   # Automated unit & integration test suite
│   ├── test_env.py          # Environment wrapper tests
│   ├── test_network.py      # Neural network & recurrent memory tests
│   └── test_perception.py   # Perception & feature extraction tests
├── submission/              # Official competition submission bundle
│   ├── agent.py             # DoomAgent entry point
│   ├── requirements.txt     # Extra pip dependencies (pinned)
│   └── weights/             # Model checkpoints tracked via Git LFS
└── validate.py              # Official evaluation validator
```

---

## 2. Global Data Types & Observation/Action Contract

All data types and dimensions strictly conform to `docs/CONTRACT.md`:

### 2.1 Observation Dict Specification

Observations are ego-centric dictionaries delivered once every 4 engine tics:

| Field | Sub-key | Shape | Data Type | Value Range | Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vitals` | — | `(17,)` | `float32` | `[-inf, inf]` | 0: health (0–200), 1: armor (0–200), 2: weapon slot (0–7), 3: weapon ammo, 4–7: ammo (bullets, shells, rockets, cells), 8–13: keys (blue, yellow, red cards; blue, yellow, red skulls) in {0, 1}, 14–16: player frame velocity (`vel_forward`, `vel_right`, `vel_up`). |
| `rays` | `distance` | `(64,)` | `float32` | `[0.0, 1.0]` | Normalized hit distance / 1024. `1.0` = nothing hit. 64 rays across 360° (ray 0 straight ahead, counter-clockwise). |
| | `hit_type` | `(64,)` | `int64` | `{0, 1, 2, 3, 4, 5}` | 0: none, 1: wall, 2: door, 3: locked door, 4: enemy, 5: item. |
| `visible_entities` | `features` | `(32, 5)` | `float32` | `[-inf, inf]` | Up to 32 nearest entities in 90° FOV & line-of-sight: `[dist / 1024, sin(rel_ang), cos(rel_ang), rel_vel_fwd, rel_vel_right]`. Zero-padded. |
| | `entity_type` | `(32,)` | `int64` | `{0..11}` | Entity category index (0 = padding, 1..3 = enemies, 4..8 = items/keys, etc.). |
| | `mask` | `(32,)` | `float32` | `{0.0, 1.0}` | `1.0` if slot contains a real entity, `0.0` if padding. |
| `rgb` (optional) | — | `(120, 160, 3)` | `uint8` | `[0, 255]` | Raw screen buffer without HUD. |
| `depth` (optional) | — | `(120, 160)` | `uint8` | `[0, 255]` | Aligned depth buffer. |

### 2.2 Action Space Specification

Actions are returned as a NumPy array of shape `(9,)` with values $\in \{0, 1\}$ (`MultiBinary(9)`):

| Index | Button Name | Action Semantics & Rules |
| :---: | :--- | :--- |
| `0` | `MOVE_FORWARD` | Move forward in player orientation |
| `1` | `MOVE_BACKWARD` | Move backward in player orientation |
| `2` | `MOVE_LEFT` | Strafe left |
| `3` | `MOVE_RIGHT` | Strafe right |
| `4` | `TURN_LEFT` | Rotate yaw left |
| `5` | `TURN_RIGHT` | Rotate yaw right |
| `6` | `ATTACK` | Fire currently selected weapon |
| `7` | `USE` | Open doors, activate switches. **Fires on button-down edge only.** Holding keeps it down; must release (`0`) for 1 step to re-trigger. Do not spam while a door is opening. |
| `8` | `SELECT_NEXT_WEAPON` | Cycle to next available weapon with ammunition. **Fires on button-down edge.** |

---

## 3. Module Specifications & Interfaces

### 3.1 Member 1: Architecture & Integration (`submission/agent.py`)

Responsible for lifecycle management, weight loading, recurrent state handling, and evaluation protocol compliance.

#### Interface: `DoomAgent`
```python
class DoomAgent:
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.MultiBinary, device: str):
        """
        Called ONCE per evaluation run.
        - Must load weights from SUBMISSION_DIR / 'weights' / 'model.pt' onto `device`.
        - Must instantiate the policy network in eval mode.
        - Must initialize recurrent state tracker to None.
        """
        ...

    def reset(self) -> None:
        """
        Called before EVERY map.
        - Must reset all recurrent hidden states (hx = None).
        - Must clear episodic memory buffers, step counters, and action trackers.
        - Must execute within 10 seconds.
        """
        ...

    def act(self, observation: dict) -> np.ndarray:
        """
        Called once per decision step (every 4 engine tics).
        - Takes unbatched observation dict conforming to CONTRACT.md.
        - Executes model inference with torch.no_grad().
        - Updates internal recurrent hidden state (hx).
        - Returns np.ndarray of shape (9,), dtype np.int8, with values in {0, 1}.
        - Must execute in < 40 ms per step.
        """
        ...
```

#### Rules for Member 1:
1. Never import dependencies not listed in `docker/requirements.lock` or `submission/requirements.txt`.
2. Do not allocate more than 2 GB VRAM on GPU.
3. Catch and safely recover from internal inference exceptions to prevent hard agent stops.

---

### 3.2 Member 2: Spatial Intelligence & Perception (`perception/`)

Responsible for computing spatial features, processing raycasts, filtering nearest threats, identifying doors and keycard pickups, and optional visual representations.

#### Functions and Classes (`perception/spatial.py`):
```python
class SpatialPerception:
    def __init__(self, config: NetworkConfig | None = None):
        ...

    def extract_spatial_features(self, obs: dict) -> dict[str, torch.Tensor]:
        """
        Convert raw gymnasium observation dict into normalized PyTorch tensors.
        
        Args:
            obs: Raw observation dict from ViZDoom environment.
        Returns:
            Dictionary containing:
              - 'vitals': Tensor (..., 17) [float32]
              - 'ray_dist': Tensor (..., 64) [float32]
              - 'ray_hit': Tensor (..., 64) [int64]
              - 'ent_feats': Tensor (..., 32, 5) [float32]
              - 'ent_type': Tensor (..., 32) [int64]
              - 'ent_mask': Tensor (..., 32) [float32]
        """
        ...

    def detect_doors(self, ray_dist: torch.Tensor, ray_hit: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Identify presence, bearing, and distance to normal and locked doors from raycasts.
        - Ray hit type 2: Normal door
        - Ray hit type 3: Locked door
        """
        ...

    def compute_threat_vector(self, ent_feats: torch.Tensor, ent_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregate ego-centric threat direction and proximity vector: shape (..., 2).
        """
        ...
```

#### Optional Visual Feature Encoder (`perception/vision.py`):
```python
class VisualEncoder(nn.Module):
    """Compact lightweight CNN for RGB / Depth buffers when visual training is enabled."""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # 4-layer conv net reducing (B, 4, 120, 160) -> (B, out_dim)
        ...

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Returns visual embedding tensor of shape (B, out_dim)."""
        ...
```

---

### 3.3 Member 3: Neural Network & Recurrent Memory (`network/`)

Responsible for the complete deep neural architecture, multimodal feature fusion, circular convolutions, threat attention, and actor-critic heads.

#### Core Model: `DoomNetwork` (`network/model.py`)
```python
class DoomNetwork(nn.Module):
    def __init__(self, config: NetworkConfig | None = None):
        ...

    def init_hidden(self, batch_size: int = 1, device: torch.device | str | None = None) -> torch.Tensor:
        """Returns zero hidden state of shape (1, batch_size, 256)."""
        ...

    def step(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> StepResult:
        """
        Rollout inference step.
        Returns StepResult(
            action: torch.Tensor (B, 9) in {0, 1},
            value: torch.Tensor (B, 1),
            log_prob: torch.Tensor (B,),
            hx: torch.Tensor (1, B, 256),
            logits: torch.Tensor (B, 9)
        )
        """
        ...

    def evaluate_actions(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> EvaluationResult:
        """
        PPO evaluation step across sequences or batches.
        Returns EvaluationResult(
            values: torch.Tensor (B, T, 1),
            log_probs: torch.Tensor (B, T),
            entropy: torch.Tensor (B, T),
            hx: torch.Tensor (1, B, 256),
            logits: torch.Tensor (B, T, 9)
        )
        """
        ...

    def get_action(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Convenience method for submission/agent.py."""
        ...
```

#### Tensor Shape Guarantees:
- `vitals`: `(B, 17)` $\to$ `(B, 64)` via `VitalsEncoder`.
- `rays`: `dist (B, 64)`, `hit (B, 64)` $\to$ `(B, 128)` via `RaycastEncoder` (circular convolution over 360° topology).
- `entities`: `feats (B, 32, 5)`, `types (B, 32)`, `mask (B, 32)` $\to$ `(B, 64)` via `EnemyAttentionModule` (guaranteed zero vector if 0 entities, strictly NaN-proof).
- `fused`: `(B, 256)` via `FeatureFusion`.
- `memory`: `(B, 256)` via GRU with hidden state `(1, B, 256)`.
- `logits`: `(B, 9)` representing independent Bernoulli parameters.
- `value`: `(B, 1)` scalar estimate of state value $V(s)$.

---

### 3.4 Member 4: Reinforcement Learning & Training (`training/`)

Responsible for PPO optimization, GAE calculation, multi-objective reward shaping, and training rollouts across maps.

#### Core Loop Interface (`training/train.py`):
```python
class RecurrentPPOTrainer:
    def __init__(self, model: DoomNetwork, env_fn, config: TrainingConfig):
        ...

    def collect_rollouts(self) -> RolloutBatch:
        """
        Execute parallel/sequential rollouts in ViZDoom environments.
        Maintains recurrent hidden states (hx) across rollout steps.
        Resets hx to init_hidden() on episode termination or truncation.
        """
        ...

    def compute_gae(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        ...

    def update_policy(self, batch: RolloutBatch) -> dict[str, float]:
        """
        Run PPO clipped surrogate update with mini-batch sequence chunking.
        Returns training metrics dict (policy_loss, value_loss, entropy, approx_kl).
        """
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Export state_dict to submission/weights/model.pt for Member 1."""
        ...
```

#### Multi-Tier Reward Shaping (`training/rewards.py`):
```python
def compute_shaped_reward(
    prev_obs: dict,
    curr_obs: dict,
    action: np.ndarray,
    env_reward: float,
    info: dict,
) -> float:
    """
    Combines:
    1. Exit milestone bonus: +10.0 on level completion.
    2. Survival & combat: +1.0 per enemy kill, -0.5 on health loss.
    3. Progression: +3.0 on keycard acquisition, +2.0 on door opening.
    4. Exploration curiosity: +0.05 per newly visited spatial cell (from ray distances).
    5. Anti-camping / stagnation penalty: -0.01 per step standing still.
    """
    ...
```

---

## 4. Integration Verification Workflow

Before merging any code to `main` or submitting:

1. **Unit Tests:** `uv run pytest tests/`
   - Checks single and batched inference.
   - Checks zero-enemy and max-enemy scenarios.
   - Checks padding invariance and NaN stability.
   - Checks hidden state persistence and resets.

2. **Official Validator:** `uv run python validate.py`
   - Verifies repository size $\le 1$ GB.
   - Verifies `DoomAgent` interface methods: `__init__`, `reset`, `act`.
   - Executes a full episode on `maps/two_rooms.wad`.
   - Confirms latency $\le 40$ ms/step on 1 CPU thread.
   - Confirms memory within 2 GB VRAM and 8 GB RAM limits.

3. **Checkpoints:** Model weights must always be saved to `submission/weights/model.pt` and tracked with Git LFS (`.gitattributes`).
