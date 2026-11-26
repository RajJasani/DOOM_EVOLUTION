# AI Architecture & Team Integration Workflow

**Author:** Member 1 — AI Architecture & Integration Engineer (Team Lead)  
**Project:** *"Can It Play DOOM?"* (Yugaantar 2026 DevForge — Track 1)  
**Contract Version:** `1.0.0-draft` (`docs/CONTRACT.md`)  
**Target Hardware Constraints:** 1 CPU thread, 8 GB RAM, 2 GB VRAM, < 40 ms/step latency  

---

## 1. Baseline Architecture & Current State

The repository currently maintains a verified integration baseline on branch `feature/neural-network`:
- **Interface Contract:** Standardized in `INTERFACES.md`.
- **Perception:** Implemented in `perception/spatial.py` (`SpatialPerception` for door detection, threat parsing, corridor clearance) with 100% passing tests in `tests/test_perception.py`.
- **Neural Network:** Implemented in `network/model.py` (`DoomNetwork` with circular 1D-CNN, NaN-proof enemy attention, GRU recurrent memory, Bernoulli actor-critic) with 100% passing tests in `tests/test_network.py`.
- **Evaluation Agent:** Implemented in `submission/agent.py` (`DoomAgent` with dynamic PyTorch model loading, recurrent hidden-state persistence, reset handling, and error fallbacks).
- **Rewards & Benchmarking:** Multi-tier reward shaping in `training/rewards.py` and episode/latency profiling in `evaluation/benchmark.py`.
- **Test Status:** **67 passed, 1 skipped** across all 68 tests in `tests/`.
- **Validation Status:** `uv run python validate.py` outputs **`PASSED`** (525 steps on `two_rooms.wad`, 0.5 ms steady-state latency).

---

## 2. Git Workflow & Branching Strategy

To prevent merge conflicts, test regressions, and accidental overwrites across 4 team members:

```
                            [origin/main or default]
                                       │
                                       ▼
                       [integration/staging-baseline]
                        (Current verified foundation)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 [feature/member2-perception] [feature/member3-network] [feature/member4-training]
  Owned by Member 2            Owned by Member 3          Owned by Member 4
  (perception/)                (network/)                 (training/)
```

### 2.1 Branch Ownership Rules
1. **Protected Branches:**
   - `main` (or default submission branch): Evaluated by the official scorer. Only Member 1 merges into `main` after Docker validation.
   - `integration/staging`: The shared integration branch where verified features are combined and tested together.
2. **Feature Branch Isolation:**
   - **Member 2 (Perception):** Branches from `integration/staging` into `feature/member2-perception`. Modifies **only** `perception/` and `tests/test_perception.py`.
   - **Member 3 (Network):** Branches from `integration/staging` into `feature/member3-network`. Modifies **only** `network/` and `tests/test_network.py`.
   - **Member 4 (Training):** Branches from `integration/staging` into `feature/member4-training`. Modifies **only** `training/` and exports checkpoints to `submission/weights/model.pt`.
3. **No Direct Pushes:** Team members must never push directly to `main` or `integration/staging`. All integration occurs via reviewed pull requests managed by Member 1.

---

## 3. Integration Acceptance Gates

Before any member's work is merged into `integration/staging`, it must satisfy its specific acceptance gate:

### Gate 2: Member 2 (Perception Delivery)
- [ ] `SpatialPerception` methods (`process_observation`, `detect_doors`, `get_nearest_threat`, `get_forward_clearance`) execute without exceptions.
- [ ] Any added helper (e.g., `detect_pickups`) returns strictly typed dictionaries or NumPy arrays.
- [ ] Observation sanitization handles 0 entities, 32 entities, and extreme distance values without raising errors.
- [ ] If optional visual CNN is added, CPU execution latency is $\le 2.0$ ms.
- [ ] `uv run pytest tests/test_perception.py` passes 100%.

### Gate 3: Member 3 (Neural Network Delivery)
- [ ] `DoomNetwork.forward()` and `DoomNetwork.step()` preserve exact tensor shapes:
  - Vitals: `(..., 17)` $\to$ `(..., 64)`
  - Rays: `(..., 64)` $\to$ `(..., 128)`
  - Visible Entities: `(..., 32, 5)` $\to$ `(..., 64)`
  - GRU Hidden State: `(1, B, 256)`
  - Action Logits: `(..., 9)`
  - Value Estimate: `(..., 1)`
- [ ] NaN-proof guarantee: Passing all-zero entity masks produces zero vectors and strictly finite non-NaN gradients during backpropagation.
- [ ] `DoomNetwork.evaluate_actions()` correctly processes chunked rollout sequences `(B, T, ...)` and applies `dones` masks to reset GRU states.
- [ ] Steady-state CPU inference latency remains $\le 1.0$ ms per step.
- [ ] Total network parameters in memory remain $\le 10$ MB.
- [ ] `uv run pytest tests/test_network.py` passes 100%.

### Gate 4: Member 4 (Training Delivery)
- [ ] `training/rollout.py` collects transitions with persistent `hx` across steps and resets `hx` on episode termination.
- [ ] `training/ppo.py` computes Generalized Advantage Estimation ($\lambda=0.95, \gamma=0.99$) without numerical divergence.
- [ ] Recurrent PPO training loop runs stably across at least 100 policy iterations without NaN losses.
- [ ] Trained checkpoint `submission/weights/model.pt` is exported via standard `torch.save(model.state_dict(), ...)` and is strictly compatible with `DoomNetwork.load_state_dict()`.
- [ ] Checkpoint file size is $< 10$ MB and properly tracked with Git LFS.

### Gate 1: Final Member 1 Integration (Production Release)
- [ ] `submission/agent.py` automatically detects and loads `submission/weights/model.pt`.
- [ ] Full test suite passes: `uv run pytest` (100% pass rate).
- [ ] Local validator passes: `uv run python validate.py` (0 errors).
- [ ] Containerized validator passes: `uv run python validate.py --docker` (0 errors).
- [ ] Total repository size including Git LFS objects is strictly $\le 1$ GB.
- [ ] No uncommitted, untracked, or `.gitignore`-masked files in `submission/`.

---

## 4. Exact Integration Verification Commands

Execute these commands in order when reviewing and merging incoming contributions:

### 4.1 Unit Test Execution
```bash
# 1. Perception unit tests (Member 2)
uv run pytest tests/test_perception.py -v

# 2. Neural network unit tests (Member 3)
uv run pytest tests/test_network.py -v

# 3. Environment wrapper unit tests
uv run pytest tests/test_env.py -v

# 4. IPC remote protocol tests
uv run pytest tests/test_remote.py -v

# 5. Validator harness unit tests
uv run pytest tests/test_validate.py -v

# 6. Complete test suite across all modules
uv run pytest
```

### 4.2 Checkpoint Loading & Agent Smoke Test
```bash
# Verify that DoomAgent loads model.pt and instantiates properly
uv run python -c "
from submission.agent import DoomAgent
from doom_env import make_env
env = make_env('maps/two_rooms.wad')
agent = DoomAgent(env.observation_space, env.action_space, 'cpu')
print('Agent Initialized:', agent.model is not None)
obs, _ = env.reset(seed=0)
action = agent.act(obs)
print('Action Shape:', action.shape, 'Values:', set(action.tolist()))
assert action.shape == (9,) and set(action.tolist()).issubset({0, 1})
print('Smoke Test PASSED')
env.close()
"
```

### 4.3 Multi-Episode Benchmark & Latency Profiling
```bash
# Profile 5 complete episodes on two_rooms.wad
uv run python evaluation/benchmark.py --wad maps/two_rooms.wad --map MAP01 --episodes 5
```

### 4.4 Official Hackathon Validation
```bash
# Local evaluation check
uv run python validate.py

# Docker containerized evaluation check (mirrors official scorer)
uv run python validate.py --docker
```

---

## 5. Member 1 Merge Review Checklist

Before approving any incoming branch merge, Member 1 must verify:

```markdown
- [ ] 1. Interface Compatibility:
      Function signatures strictly match INTERFACES.md. No unauthorized parameter additions.
- [ ] 2. Tensor Dimension Alignment:
      Tensors strictly match CONTRACT.md (vitals: 17, rays: 64, entities: 32x5, actions: 9).
- [ ] 3. Data Types & Values:
      Actions are np.int8 in {0, 1}; observations are float32 / int64 as specified.
- [ ] 4. Dependency Isolation:
      No new pip dependencies added to pyproject.toml without explicit Member 1 approval.
      Only allowed dependencies are in docker/requirements.lock.
- [ ] 5. Test Regression Check:
      `uv run pytest` passes 100% of all 68 tests. No previously passing tests broken.
- [ ] 6. Runtime Performance:
      Steady-state latency remains < 5.0 ms/step (far below the 40 ms budget).
      First-step warm-up latency remains < 2.0 s (limit 10 s).
- [ ] 7. Clean Directory Scope:
      Member 2 only touched perception/ and test_perception.py.
      Member 3 only touched network/ and test_network.py.
      Member 4 only touched training/ and submission/weights/model.pt.
```

---

## 6. Rollback & Recovery Precautions

1. **Working Tree Protection:**
   - Never run `git reset --hard` or `git clean -fd` without creating a backup stash or local backup branch (`git branch backup-$(date +%s)`).
2. **Isolated Validation Workspace:**
   - Test incoming contributions on a dedicated local branch before merging to `integration/staging`.
   - If an incoming change causes regressions, immediately revert the PR branch:
     ```bash
     git checkout integration/staging
     git branch -D feature/faulty-branch
     ```
3. **Weights Rollback:**
   - If a newly trained `model.pt` exhibits reward collapse or high latency, restore the previous stable checkpoint or fall back to initialized weights to ensure `validate.py` remains in a passing state.
4. **Git LFS Safety:**
   - Always verify that checkpoints are tracked via Git LFS:
     ```bash
     git lfs ls-files
     ```
   - Never commit `.pt` files without `.gitattributes` tracking active.
