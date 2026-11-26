"""Comprehensive Unit Test Suite for DOOM Agent Neural Network & Memory Architecture.

Implements all 10 testing requirements from Member 3 specification:
  1. Single observation inference
  2. Batched observation inference
  3. Sequential observations (PPO BPTT rollouts)
  4. Different numbers of visible enemies
  5. Empty enemy lists (0 visible entities, no NaNs)
  6. Padding and attention masking invariance
  7. Hidden-state initialization
  8. Hidden-state persistence across steps
  9. Hidden-state reset between maps and episode boundaries
  10. Output dimensions, numerical stability, and latency constraints
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure repository root and env/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "env"))

from doom_env import spaces as S
from network import (
    DoomNetwork,
    NetworkConfig,
    VitalsEncoder,
    RaycastEncoder,
    EnemyAttentionModule,
    RecurrentMemory,
    PolicyHead,
    ValueHead,
)


def create_synthetic_observation(
    batch_size: int | None = None,
    seq_len: int | None = None,
    num_visible_enemies: int = 4,
    device: str = "cpu",
) -> dict:
    """Helper to generate valid DOOM observations conforming to CONTRACT.md."""
    is_batched = batch_size is not None
    is_seq = seq_len is not None

    shape_prefix = ()
    if is_batched:
        shape_prefix = (batch_size, seq_len) if is_seq else (batch_size,)

    # Vitals: 17 scalar features
    vitals = torch.zeros(*shape_prefix, 17, dtype=torch.float32, device=device)
    vitals[..., 0] = 100.0  # health
    vitals[..., 1] = 50.0   # armor
    vitals[..., 2] = 2.0    # selected weapon
    vitals[..., 3] = 25.0   # weapon ammo
    vitals[..., 4:8] = torch.tensor([50.0, 16.0, 0.0, 0.0], device=device) # ammo
    vitals[..., 8] = 1.0    # blue key card
    vitals[..., 14] = 8.5   # vel_forward
    vitals[..., 15] = -2.0  # vel_right

    # Rays: 64 distance, 64 hit types (0..5)
    rays_dist = torch.rand(*shape_prefix, 64, dtype=torch.float32, device=device)
    rays_type = torch.randint(0, 6, (*shape_prefix, 64), dtype=torch.long, device=device)

    # Entities: MAX_ENTITIES=32
    ent_feats = torch.zeros(*shape_prefix, 32, 5, dtype=torch.float32, device=device)
    ent_type = torch.zeros(*shape_prefix, 32, dtype=torch.long, device=device)
    ent_mask = torch.zeros(*shape_prefix, 32, dtype=torch.float32, device=device)

    k = min(num_visible_enemies, 32)
    if k > 0:
        ent_feats[..., :k, 0] = torch.rand(*shape_prefix, k, device=device) * 2.0  # distance
        ent_feats[..., :k, 1] = torch.sin(torch.rand(*shape_prefix, k, device=device) * 1.5 - 0.75)
        ent_feats[..., :k, 2] = torch.cos(torch.rand(*shape_prefix, k, device=device) * 1.5 - 0.75)
        ent_feats[..., :k, 3] = torch.randn(*shape_prefix, k, device=device) * 5.0
        ent_feats[..., :k, 4] = torch.randn(*shape_prefix, k, device=device) * 5.0
        ent_type[..., :k] = torch.randint(1, 4, (*shape_prefix, k), dtype=torch.long, device=device)  # enemy types
        ent_mask[..., :k] = 1.0

    return {
        "vitals": vitals,
        "rays": {
            "distance": rays_dist,
            "hit_type": rays_type,
        },
        "visible_entities": {
            "features": ent_feats,
            "entity_type": ent_type,
            "mask": ent_mask,
        },
    }


# ==============================================================================
# Requirement 1: Single observation inference
# ==============================================================================
def test_single_observation_inference():
    """Verify single observation inference produces valid actions and values."""
    model = DoomNetwork()
    obs = create_synthetic_observation()

    # 1. step() execution
    res = model.step(obs, hx=None, deterministic=False)
    assert res.action.shape == (9,)
    assert set(res.action.tolist()).issubset({0.0, 1.0})
    assert res.value.shape == (1,)
    assert res.log_prob.shape == ()
    assert res.hx.shape == (1, 1, 256)

    # 2. get_action() execution (official agent interface)
    action_np, next_hx = model.get_action(obs, hx=None, deterministic=True)
    assert isinstance(action_np, np.ndarray)
    assert action_np.shape == (9,)
    assert action_np.dtype == np.int8
    assert ((action_np == 0) | (action_np == 1)).all()
    assert next_hx.shape == (1, 1, 256)

    # 3. Test with official gymnasium sampled observation
    gym_obs = S.observation_space().sample()
    action_gym, hx_gym = model.get_action(gym_obs, hx=None)
    assert action_gym.shape == (9,)
    assert action_gym.dtype == np.int8


# ==============================================================================
# Requirement 2: Batched observation inference
# ==============================================================================
def test_batched_observation_inference():
    """Verify batched observation forward pass and gradient flow."""
    model = DoomNetwork()
    batch_size = 8
    obs = create_synthetic_observation(batch_size=batch_size)

    logits, values, next_hx = model(obs, hx=None)
    assert logits.shape == (batch_size, 9)
    assert values.shape == (batch_size, 1)
    assert next_hx.shape == (1, batch_size, 256)

    # Check backpropagation
    loss = logits.sum() + values.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"


# ==============================================================================
# Requirement 3: Sequential observations (PPO BPTT rollouts)
# ==============================================================================
def test_sequential_observations():
    """Verify multi-step sequential rollout and evaluate_actions for PPO."""
    model = DoomNetwork()
    batch_size = 4
    seq_len = 10
    obs_seq = create_synthetic_observation(batch_size=batch_size, seq_len=seq_len)
    actions = torch.randint(0, 2, (batch_size, seq_len, 9)).float()
    dones = torch.zeros(batch_size, seq_len)
    dones[:, 5] = 1.0  # episode boundary midway

    init_hx = model.init_hidden(batch_size=batch_size)
    eval_res = model.evaluate_actions(obs_seq, hx=init_hx, actions=actions, dones=dones)

    assert eval_res.values.shape == (batch_size, seq_len, 1)
    assert eval_res.log_probs.shape == (batch_size, seq_len)
    assert eval_res.entropy.shape == (batch_size, seq_len)
    assert eval_res.hx.shape == (1, batch_size, 256)

    # Check loss and gradient backpropagation through time
    loss = eval_res.values.mean() + eval_res.log_probs.mean() + eval_res.entropy.mean()
    loss.backward()
    assert not torch.isnan(model.actor.actor_net[0].weight.grad).any()


# ==============================================================================
# Requirement 4: Different numbers of visible enemies
# ==============================================================================
@pytest.mark.parametrize("num_enemies", [1, 2, 5, 16, 32])
def test_different_numbers_of_visible_enemies(num_enemies):
    """Verify attention module cleanly handles varying entity counts."""
    model = DoomNetwork()
    obs = create_synthetic_observation(batch_size=2, num_visible_enemies=num_enemies)

    logits, values, _ = model(obs)
    assert logits.shape == (2, 9)
    assert values.shape == (2, 1)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(values).any()


# ==============================================================================
# Requirement 5: Empty enemy lists (0 visible entities)
# ==============================================================================
def test_empty_enemy_lists():
    """Verify attention module handles 0 visible entities without NaN collapse."""
    model = DoomNetwork()
    # Batch where all samples have 0 entities
    obs_empty = create_synthetic_observation(batch_size=4, num_visible_enemies=0)
    assert obs_empty["visible_entities"]["mask"].sum() == 0

    logits, values, _ = model(obs_empty)
    assert not torch.isnan(logits).any(), "Logits contain NaN with empty enemies"
    assert not torch.isnan(values).any(), "Values contain NaN with empty enemies"

    # Mixed batch: sample 0 has 0 enemies, sample 1 has 3 enemies
    obs_mixed = create_synthetic_observation(batch_size=2, num_visible_enemies=0)
    obs_mixed["visible_entities"]["mask"][1, :3] = 1.0
    obs_mixed["visible_entities"]["entity_type"][1, :3] = 1

    logits_m, values_m, _ = model(obs_mixed)
    assert not torch.isnan(logits_m).any()
    assert not torch.isnan(values_m).any()


# ==============================================================================
# Requirement 6: Padding and attention masking invariance
# ==============================================================================
def test_padding_and_attention_masking():
    """Verify that values in padded entity slots (mask=0) do not alter predictions."""
    model = DoomNetwork()
    obs1 = create_synthetic_observation(batch_size=2, num_visible_enemies=2)

    # Clone obs1 to obs2 but inject completely different arbitrary garbage in padded slots (indices >= 2)
    obs2 = {
        "vitals": obs1["vitals"].clone(),
        "rays": {
            "distance": obs1["rays"]["distance"].clone(),
            "hit_type": obs1["rays"]["hit_type"].clone(),
        },
        "visible_entities": {
            "features": obs1["visible_entities"]["features"].clone(),
            "entity_type": obs1["visible_entities"]["entity_type"].clone(),
            "mask": obs1["visible_entities"]["mask"].clone(),
        },
    }
    # Corrupt padded slots in obs2
    obs2["visible_entities"]["features"][:, 2:, :] = 999.0
    obs2["visible_entities"]["entity_type"][:, 2:] = 11

    with torch.no_grad():
        logits1, values1, _ = model(obs1)
        logits2, values2, _ = model(obs2)

    # Outputs MUST be identical because padded slots are strictly masked out
    torch.testing.assert_close(logits1, logits2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values1, values2, atol=1e-5, rtol=1e-5)


# ==============================================================================
# Requirement 7: Hidden-state initialization
# ==============================================================================
def test_hidden_state_initialization():
    """Verify hidden state initialization dimensions and zeros."""
    model = DoomNetwork()
    hx = model.init_hidden(batch_size=5)
    assert hx.shape == (1, 5, 256)
    assert (hx == 0.0).all()
    assert hx.device == model.device


# ==============================================================================
# Requirement 8: Hidden-state persistence across steps
# ==============================================================================
def test_hidden_state_persistence():
    """Verify hidden state evolves sequentially across consecutive steps."""
    model = DoomNetwork()
    hx0 = model.init_hidden(batch_size=1)

    obs1 = create_synthetic_observation(batch_size=1)
    res1 = model.step(obs1, hx=hx0)
    hx1 = res1.hx

    # Hidden state must update and differ from zeros
    assert not torch.allclose(hx0, hx1)

    obs2 = create_synthetic_observation(batch_size=1)
    res2 = model.step(obs2, hx=hx1)
    hx2 = res2.hx

    # Step 2 hidden state must carry forward information
    assert not torch.allclose(hx1, hx2)

    # Comparing step2 run with persistent hx1 vs blank hx0 should produce different decisions
    res2_blank = model.step(obs2, hx=hx0)
    assert not torch.allclose(res2.logits, res2_blank.logits)


# ==============================================================================
# Requirement 9: Hidden-state reset
# ==============================================================================
def test_hidden_state_reset():
    """Verify that resetting hidden state cleanly resets memory between maps."""
    model = DoomNetwork()
    obs = create_synthetic_observation(batch_size=1)

    # Warm up recurrent memory
    hx = model.init_hidden(batch_size=1)
    for _ in range(5):
        _, hx = model.get_action(obs, hx=hx)

    assert not torch.allclose(hx, torch.zeros_like(hx))

    # Reset memory (as done between maps in agent.reset())
    hx_reset = model.init_hidden(batch_size=1)
    assert (hx_reset == 0.0).all()

    # Step from reset state matches an initial step identically
    res_fresh = model.step(obs, hx=None)
    res_reset = model.step(obs, hx=hx_reset)
    torch.testing.assert_close(res_fresh.logits, res_reset.logits)
    torch.testing.assert_close(res_fresh.value, res_reset.value)


# ==============================================================================
# Requirement 10: Output dimensions, numerical stability & latency
# ==============================================================================
def test_output_dimensions_numerical_stability_and_latency():
    """Verify numerical stability on extreme inputs and test inference latency."""
    import time

    model = DoomNetwork()

    # Extreme observation values
    obs_extreme = create_synthetic_observation(batch_size=2)
    obs_extreme["vitals"][:, 0] = 0.0       # 0 health (near death)
    obs_extreme["vitals"][:, 1] = 200.0     # max armor
    obs_extreme["vitals"][:, 14:17] = 200.0 # high velocities
    obs_extreme["rays"]["distance"][:, :] = 0.0 # wall right in face
    obs_extreme["rays"]["hit_type"][:, :] = 3   # locked doors everywhere

    logits, values, _ = model(obs_extreme)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(values).any()
    assert not torch.isinf(values).any()

    # Measure single-step inference latency
    obs_single = create_synthetic_observation()
    hx = model.init_hidden(batch_size=1)

    # Warmup
    for _ in range(10):
        _ = model.step(obs_single, hx=hx)

    # Time 50 steps
    start_time = time.perf_counter()
    n_steps = 50
    for _ in range(n_steps):
        res = model.step(obs_single, hx=hx)
        hx = res.hx
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / n_steps

    print(f"\nAverage single-step latency: {elapsed_ms:.2f} ms (Budget limit: 40.0 ms)")
    # Must comfortably meet the 40 ms per-step compute budget
    assert elapsed_ms < 40.0, f"Latency {elapsed_ms:.2f} ms exceeds 40 ms limit"



# ==============================================================================
# Live Environment Integration: Real Observations from ViZDoom
# ==============================================================================
def test_live_doom_env_episode():
    """Verify DoomNetwork interacting with a live headless ViZDoom environment."""
    from doom_env import make_env

    wad_path = REPO_ROOT / "maps" / "two_rooms.wad"
    if not wad_path.is_file():
        pytest.skip(f"Map file {wad_path} not found.")

    env = make_env(str(wad_path), "MAP01", episode_ticks=200, render_buffers=False)
    obs, info = env.reset(seed=42)

    model = DoomNetwork()
    hx = model.init_hidden(batch_size=1)

    steps_taken = 0
    total_reward = 0.0

    for step in range(25):
        # 1. Step model on actual live observation dict from ViZDoom
        action_np, hx = model.get_action(obs, hx=hx, deterministic=True)
        assert action_np.shape == (9,)
        assert action_np.dtype == np.int8

        # 2. Step environment with model's action
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        steps_taken += 1
        obs = next_obs

        if terminated or truncated:
            break

    assert steps_taken == 25
    assert hx.shape == (1, 1, 256)
    env.close()
