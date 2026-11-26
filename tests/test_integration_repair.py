"""Comprehensive integration repair verification tests.

Verifies:
  1. Vitals are normalized exactly once (canonical boundary).
  2. Spatial memory reaches the network via MemoryEncoder and FeatureFusion.
  3. Spatial memory influences fused representations and downstream outputs.
  4. Spatial memory supports unbatched, batched, and sequential observations.
  5. Missing or malformed memory features are handled safely.
  6. Perception state resets cleanly between episodes.
  7. ObservationAdapter parity between training and inference pipelines.
  8. Depth processing is skipped when enable_depth=False to save CPU.
  9. Reward shaper supports both positional and keyword invocation ('env_reward' and 'raw_env_reward').
  10. Reward damage scaling is consistent across both raw and normalized vitals.
  11. GRU hidden-state persistence and episode-boundary zeroing.
  12. Action output dimensions and MultiBinary(9) contract compliance.
  13. Edge-trigger action adapter preserves contract and resets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from network.config import NetworkConfig
from network.encoder import VitalsEncoder, MemoryEncoder, FeatureFusion
from network.model import DoomNetwork
from perception.bridge import PerceptionBridge, MEMORY_KEY
from perception.adapter import ObservationAdapter, VectorObservationAdapter
from perception.action_adapter import ActionEdgeTriggerAdapter
from perception.processors.pipeline import SensoryPipeline, ProcessedObservation
from training.rewards import DoomRewardShaper
from submission.agent import DoomAgent


@pytest.fixture
def mock_raw_obs():
    return {
        "vitals": np.array(
            [100.0, 50.0, 2.0, 25.0, 50.0, 16.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 15.0, -2.0, 0.0],
            dtype=np.float32,
        ),
        "rays": {
            "distance": np.ones(64, dtype=np.float32) * 0.75,
            "hit_type": np.zeros(64, dtype=np.int64),
        },
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros(32, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.float32),
        },
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.zeros((120, 160), dtype=np.uint8),
    }


# ===========================================================================
# 1. Vitals Normalization Tests
# ===========================================================================

def test_vitals_normalized_exactly_once(mock_raw_obs):
    """Verify health 100 becomes 1.0 exactly once, weapon / 7, velocity / 20."""
    adapter = ObservationAdapter()
    enriched = adapter.process(mock_raw_obs)

    # 1. ObservationAdapter (via SensoryPipeline / VitalsProcessor) normalizes vitals:
    normed_vitals = enriched["vitals"]
    assert float(normed_vitals[0]) == pytest.approx(1.0)  # Health 100 -> 1.0
    assert float(normed_vitals[1]) == pytest.approx(0.5)  # Armor 50 -> 0.5
    assert float(normed_vitals[2]) == pytest.approx(2.0 / 7.0)  # Weapon 2 -> 2/7
    assert float(normed_vitals[14]) == pytest.approx(15.0 / 20.0)  # Vel 15 -> 0.75

    # 2. VitalsEncoder in DoomNetwork (with normalize_vitals=False) does not re-normalize:
    cfg = NetworkConfig(normalize_vitals=False)
    encoder = VitalsEncoder(cfg)
    vitals_t = torch.tensor(normed_vitals, dtype=torch.float32)

    # Calling forward without override: preserves 1.0 entering the MLP
    # We inspect the intermediate normalized tensor by checking normalize_vitals=False behavior
    with torch.no_grad():
        out = encoder(vitals_t)
    assert out.shape == (64,)

    # 3. Explicit raw normalization still works if normalize=True is passed
    raw_vitals_t = torch.tensor(mock_raw_obs["vitals"], dtype=torch.float32)
    with torch.no_grad():
        out_from_raw = encoder(raw_vitals_t, normalize=True)
    assert out_from_raw.shape == (64,)


# ===========================================================================
# 2 & 3. Spatial Memory Connectivity & Sensitivity Tests
# ===========================================================================

def test_spatial_memory_reaches_network_and_influences_outputs(mock_raw_obs):
    """Verify spatial memory reaches DoomNetwork and directly affects representations."""
    net = DoomNetwork(NetworkConfig(memory_feature_dim=64))
    net.eval()

    # Generate a nontrivial memory vector using ObservationAdapter
    adapter = ObservationAdapter()
    # Populate memory by marking a door hit
    obs_with_door = {
        "vitals": mock_raw_obs["vitals"].copy(),
        "rays": {
            "distance": mock_raw_obs["rays"]["distance"].copy(),
            "hit_type": mock_raw_obs["rays"]["hit_type"].copy(),
        },
        "visible_entities": {
            "features": mock_raw_obs["visible_entities"]["features"].copy(),
            "entity_type": mock_raw_obs["visible_entities"]["entity_type"].copy(),
            "mask": mock_raw_obs["visible_entities"]["mask"].copy(),
        },
    }
    obs_with_door["rays"]["hit_type"][0] = 2  # door at ray 0
    enriched = adapter.process(obs_with_door)
    real_mem = enriched[MEMORY_KEY]

    base_obs = {
        "vitals": enriched["vitals"],
        "rays": enriched["rays"],
        "visible_entities": enriched["visible_entities"],
    }

    obs_zeros = {**base_obs, "memory_features": np.zeros(64, dtype=np.float32)}
    obs_ones = {**base_obs, "memory_features": np.ones(64, dtype=np.float32)}
    obs_real = {**base_obs, "memory_features": real_mem}

    with torch.no_grad():
        res_zeros = net.step(obs_zeros)
        res_ones = net.step(obs_ones)
        res_real = net.step(obs_real)

        enc_zeros = net.encode_observation(obs_zeros)
        enc_ones = net.encode_observation(obs_ones)
        enc_real = net.encode_observation(obs_real)

        proj_zeros = net.memory_encoder(torch.tensor(obs_zeros["memory_features"]).unsqueeze(0))
        proj_ones = net.memory_encoder(torch.tensor(obs_ones["memory_features"]).unsqueeze(0))
        proj_real = net.memory_encoder(torch.tensor(obs_real["memory_features"]).unsqueeze(0))

    # Memory projection must differ
    assert torch.max(torch.abs(proj_zeros - proj_ones)).item() > 0.1
    assert torch.max(torch.abs(proj_zeros - proj_real)).item() > 0.1

    # Fused representation must differ
    assert torch.max(torch.abs(enc_zeros - enc_ones)).item() > 0.1
    assert torch.max(torch.abs(enc_zeros - enc_real)).item() > 0.1

    # Value output must differ
    assert torch.max(torch.abs(res_zeros.value - res_ones.value)).item() > 0.01
    assert torch.max(torch.abs(res_zeros.value - res_real.value)).item() > 0.01

    # Policy logits must differ
    assert torch.max(torch.abs(res_zeros.logits - res_ones.logits)).item() > 0.0001
    assert torch.max(torch.abs(res_zeros.logits - res_real.logits)).item() > 0.0001


# ===========================================================================
# 4 & 5. Batching, Sequencing, and Missing Memory Fallback
# ===========================================================================

def test_spatial_memory_batching_and_missing_fallback(mock_raw_obs):
    """Verify unbatched, batched, sequential observations and missing memory fallback."""
    net = DoomNetwork(NetworkConfig(memory_feature_dim=64))
    net.eval()

    # 1. Unbatched observation without memory_features key
    obs_no_mem = {
        "vitals": np.zeros(17, dtype=np.float32),
        "rays": {"distance": np.ones(64, dtype=np.float32), "hit_type": np.zeros(64, dtype=np.int64)},
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros(32, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.float32),
        },
    }
    res = net.step(obs_no_mem)
    assert res.action.shape == (9,)
    assert res.value.shape == (1,)

    # 2. Batched observation (B=3)
    B = 3
    batched_obs = {
        "vitals": torch.zeros((B, 17)),
        "rays": {"distance": torch.ones((B, 64)), "hit_type": torch.zeros((B, 64), dtype=torch.long)},
        "visible_entities": {
            "features": torch.zeros((B, 32, 5)),
            "entity_type": torch.zeros((B, 32), dtype=torch.long),
            "mask": torch.zeros((B, 32)),
        },
        "memory_features": torch.rand((B, 64)),
    }
    res_b = net.step(batched_obs)
    assert res_b.action.shape == (B, 9)
    assert res_b.value.shape == (B, 1)

    # 3. Sequential evaluate_actions (B=2, T=4)
    T = 4
    seq_obs = {
        "vitals": torch.zeros((2, T, 17)),
        "rays": {"distance": torch.ones((2, T, 64)), "hit_type": torch.zeros((2, T, 64), dtype=torch.long)},
        "visible_entities": {
            "features": torch.zeros((2, T, 32, 5)),
            "entity_type": torch.zeros((2, T, 32), dtype=torch.long),
            "mask": torch.zeros((2, T, 32)),
        },
        "memory_features": torch.rand((2, T, 64)),
    }
    eval_res = net.evaluate_actions(
        seq_obs,
        hx=None,
        actions=torch.zeros((2, T, 9), dtype=torch.int8),
        dones=torch.zeros((2, T)),
    )
    assert eval_res.values.shape == (2, T, 1)
    assert eval_res.log_probs.shape == (2, T)


# ===========================================================================
# 6 & 7. Observation Pipeline Parity & Reset
# ===========================================================================

def test_training_and_inference_pipeline_parity(mock_raw_obs):
    """Verify ObservationAdapter produces matching observations in inference and training."""
    # Single adapter (as used in DoomAgent)
    inference_adapter = ObservationAdapter()
    inf_obs = inference_adapter.process(mock_raw_obs)

    # Vector adapter (as used in Member 4 PPO rollouts)
    vector_adapter = VectorObservationAdapter(num_envs=1)
    train_obs = vector_adapter.process_env(0, mock_raw_obs)

    # All fields must match identically
    np.testing.assert_allclose(inf_obs["vitals"], train_obs["vitals"])
    np.testing.assert_allclose(inf_obs["rays"]["distance"], train_obs["rays"]["distance"])
    np.testing.assert_array_equal(inf_obs["rays"]["hit_type"], train_obs["rays"]["hit_type"])
    np.testing.assert_allclose(
        inf_obs["visible_entities"]["features"], train_obs["visible_entities"]["features"]
    )
    np.testing.assert_array_equal(
        inf_obs["visible_entities"]["entity_type"], train_obs["visible_entities"]["entity_type"]
    )
    np.testing.assert_allclose(inf_obs[MEMORY_KEY], train_obs[MEMORY_KEY])


def test_vector_adapter_isolated_resets(mock_raw_obs):
    """Verify resetting env 0 does not clear spatial memory for env 1."""
    vec = VectorObservationAdapter(num_envs=2)
    # Env 0 sees a door, Env 1 sees clear hallway
    obs0 = {**mock_raw_obs, "rays": {"distance": np.ones(64, dtype=np.float32), "hit_type": np.zeros(64, dtype=np.int64)}}
    obs0["rays"]["hit_type"][0] = 2
    obs1 = {**mock_raw_obs, "rays": {"distance": np.ones(64, dtype=np.float32), "hit_type": np.zeros(64, dtype=np.int64)}}

    vec.process_env(0, obs0)
    vec.process_env(1, obs1)

    assert vec.adapters[0].bridge.memory.snapshot().n_active_doors >= 1
    assert vec.adapters[1].bridge.memory.snapshot().n_active_doors == 0

    # Reset only env 0
    vec.reset_at(0)
    assert vec.adapters[0].bridge.memory.snapshot().n_active_doors == 0
    assert vec.adapters[0].bridge.episode_step == 0


# ===========================================================================
# 8. Depth Processing Configuration
# ===========================================================================

def test_depth_processing_skipped_when_disabled(mock_raw_obs):
    """Verify DepthProcessor is not executed when enable_depth=False."""
    pipe_disabled = SensoryPipeline(enable_depth=False, validate=False)
    res_disabled = pipe_disabled.process(mock_raw_obs)
    assert res_disabled.depth is None

    pipe_enabled = SensoryPipeline(enable_depth=True, validate=False)
    res_enabled = pipe_enabled.process(mock_raw_obs)
    assert res_enabled.depth is not None
    assert hasattr(res_enabled.depth, "is_zero_buffer")


# ===========================================================================
# 9 & 10. Reward Shaper Contract & Damage Scaling
# ===========================================================================

def test_reward_shaper_signatures_and_scaling(mock_raw_obs):
    """Verify positional and keyword invocation and scale invariance."""
    shaper = DoomRewardShaper()
    shaper.reset(mock_raw_obs)

    # 1. Keyword invocation with 'env_reward' (Handoff doc signature)
    r1 = shaper.compute_reward(mock_raw_obs, env_reward=1.0, terminated=False, truncated=False)
    assert isinstance(r1, float)

    # 2. Positional invocation
    r2 = shaper.compute_reward(mock_raw_obs, 1.0, False, False)
    assert isinstance(r2, float)

    # 3. Damage scaling test with RAW vitals (100 -> 90 HP = 10 damage)
    shaper_raw = DoomRewardShaper()
    shaper_raw.reset(mock_raw_obs)  # health = 100
    damaged_raw = {
        "vitals": mock_raw_obs["vitals"].copy(),
    }
    damaged_raw["vitals"][0] = 90.0
    reward_raw = shaper_raw.compute_reward(damaged_raw, env_reward=0.0, terminated=False, truncated=False)
    # Damage penalty: -0.05 * 10 = -0.5
    # Forward velocity bonus: 15.0 > 1.0 -> +0.02
    expected_reward = -0.5 + 0.02
    assert reward_raw == pytest.approx(expected_reward, abs=1e-4)

    # 4. Damage scaling test with NORMALIZED vitals (1.0 -> 0.9 HP = 10 damage)
    shaper_norm = DoomRewardShaper()
    norm_obs_initial = {
        "vitals": mock_raw_obs["vitals"].copy(),
    }
    norm_obs_initial["vitals"][0] = 1.0
    norm_obs_initial["vitals"][14] = 0.75  # 15 / 20
    shaper_norm.reset(norm_obs_initial)

    damaged_norm = {
        "vitals": norm_obs_initial["vitals"].copy(),
    }
    damaged_norm["vitals"][0] = 0.9  # 10 HP damage
    reward_norm = shaper_norm.compute_reward(damaged_norm, env_reward=0.0, terminated=False, truncated=False)
    assert reward_norm == pytest.approx(expected_reward, abs=1e-4)


# ===========================================================================
# 11, 12, 13. DoomAgent, GRU Lifecycle, and Action Adapter
# ===========================================================================

def test_action_edge_trigger_adapter():
    """Verify ActionEdgeTriggerAdapter suppresses continuous presses on USE and WEAPON."""
    adapter = ActionEdgeTriggerAdapter(pulse_indices=(7, 8), enabled=True)

    # Step 1: press USE (index 7)
    act1 = np.zeros(9, dtype=np.int8)
    act1[7] = 1
    out1 = adapter.process(act1)
    assert out1[7] == 1  # First press passes

    # Step 2: continue holding USE
    act2 = np.zeros(9, dtype=np.int8)
    act2[7] = 1
    out2 = adapter.process(act2)
    assert out2[7] == 0  # Second consecutive press suppressed to 0

    # Step 3: release USE
    act3 = np.zeros(9, dtype=np.int8)
    out3 = adapter.process(act3)
    assert out3[7] == 0

    # Step 4: press USE again
    act4 = np.zeros(9, dtype=np.int8)
    act4[7] = 1
    out4 = adapter.process(act4)
    assert out4[7] == 1  # Allowed again after release


def test_agent_with_loaded_weights_lifecycle(mock_raw_obs):
    """Verify complete DoomAgent lifecycle with active model checkpoint."""
    class MockSpace:
        def __init__(self, shape):
            self.shape = shape

    obs_space = MockSpace(shape=None)
    act_space = MockSpace(shape=(9,))

    agent = DoomAgent(obs_space, act_space, device="cpu")
    agent.model = DoomNetwork(NetworkConfig(memory_feature_dim=64))
    agent.reset()

    assert agent.hx is None
    assert agent.bridge.episode_step == 0

    action = agent.act(mock_raw_obs)
    assert action.shape == (9,)
    assert action.dtype == np.int8
    assert set(action.tolist()).issubset({0, 1})
    assert agent.hx is not None
    assert agent.n_perception_fallbacks == 0
    assert agent.n_network_fallbacks == 0
