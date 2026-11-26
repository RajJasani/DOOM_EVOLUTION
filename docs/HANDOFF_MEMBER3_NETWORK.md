# Member 3 Handoff: Neural Network & Memory Architecture

**Target Role:** Member 3 (Neural Network & Memory)  
**Lead Coordinator:** Member 1 (AI Architecture & Integration)  
**Target Module:** `network/`  
**Reference Specs:** `docs/CONTRACT.md`, `docs/ARCHITECTURE_MEMBER3.md`, `INTERFACES.md`  

---

## 1. Current Implementation Status

`network/` contains an operational deep neural network that has been fully integrated into `submission/agent.py` and validated under `validate.py`:
- **`VitalsEncoder`**: 2-layer MLP (17 $\to$ 64 $\to$ 64) with component-wise scaling for health, armor, ammo, keys, velocities.
- **`RaycastEncoder`**: Circular 1D convolution over 64 ego-centric rays + 16-dim hit-type embedding $\to$ 128 latent dimensions.
- **`EnemyAttentionModule`**: Multi-head self-attention and cross-attention over up to 32 entities with NaN-proof masking $\to$ 64 latent dimensions.
- **`FeatureFusion`**: Concatenates encoders ($64 + 128 + 64 = 256$) $\to$ LayerNorm + GELU $\to$ 256.
- **`RecurrentMemory`**: 1-layer GRU (hidden size 256) tracking long-horizon spatial memory.
- **`PolicyHead` & `ValueHead`**: MultiBinary(9) Bernoulli action logits and scalar value estimate $V(s)$.

All 15 tests in `tests/test_network.py` pass. Steady-state CPU inference runs in **0.5 ms** (budget: 40 ms).

---

## 2. Exact Forward-Pass & Inference Interfaces

### 2.1 Model Initialization
```python
model = DoomNetwork(config=NetworkConfig())
```
Hidden state initialization:
```python
hx = model.init_hidden(batch_size=B, device=device)  # Returns torch.Tensor of shape (1, B, 256)
```

### 2.2 Inference Step (`submission/agent.py` & Rollouts)
```python
step_res = model.step(obs, hx=hx, deterministic=False)
```
- **Inputs:**
  - `obs`: Unbatched observation dict (or batched `(B, ...)`).
  - `hx`: `(1, B, 256)` or `None`.
  - `deterministic`: `bool` (set `True` during evaluation greedy actions, `False` during training exploration).
- **Return Value:** `StepResult` namedtuple:
  - `step_res.action`: `torch.Tensor` of shape `(B, 9)` or `(9,)` with values in $\{0, 1\}$.
  - `step_res.value`: `torch.Tensor` of shape `(B, 1)` or `(1,)`.
  - `step_res.log_prob`: `torch.Tensor` of shape `(B,)` or scalar.
  - `step_res.hx`: `torch.Tensor` of shape `(1, B, 256)` (the updated recurrent state).
  - `step_res.logits`: `torch.Tensor` of shape `(B, 9)` or `(9,)`.

### 2.3 Evaluation Step (`training/ppo.py` Updates)
```python
eval_res = model.evaluate_actions(obs, hx, actions, dones=None)
```
- **Inputs:**
  - `obs`: Batched observation sequences of shape `(B, T, ...)`.
  - `hx`: Recurrent state at step 0 of each sequence `(1, B, 256)`.
  - `actions`: Executed actions tensor of shape `(B, T, 9)`.
  - `dones`: Sequence termination mask tensor of shape `(B, T)`.
- **Return Value:** `EvaluationResult` namedtuple:
  - `eval_res.values`: `(B, T, 1)` state-value estimates.
  - `eval_res.log_probs`: `(B, T)` action log probabilities under current policy.
  - `eval_res.entropy`: `(B, T)` action distribution entropy.
  - `eval_res.hx`: `(1, B, 256)` final hidden state.

### 2.4 Agent Convenience Interface (`submission/agent.py`)
```python
action_np, next_hx = model.get_action(obs, hx=hx, deterministic=True)
```
- Returns `action_np: np.ndarray` of shape `(9,)`, dtype `int8`, values in $\{0, 1\}$.

---

## 3. Pending Deliverables & Tasks for Member 3

1. **BPTT Sequence Packing & Masking Verification:**
   - Verify with Member 4 that passing chunked sequences `(B, T, ...)` through `evaluate_actions` with `dones` correctly resets recurrent memory across episode boundaries inside rollout chunks.
2. **Numerical & Gradient Stability:**
   - Confirm that backpropagation through `EnemyAttentionModule` produces finite gradients across sequences with zero visible threats (`mask.sum() == 0`).
3. **Weight Export & Loading Verification:**
   - Provide a utility to verify checkpoint integrity:
     ```python
     def verify_checkpoint(path: Path) -> bool:
         # Load state_dict, compare layer names, shapes, and verify no NaNs
     ```
4. **VRAM & Memory Assurance:**
   - Ensure total model parameters remain under 10 MB in memory (currently ~1.2 MB), well within the 2 GB VRAM per-agent budget.
