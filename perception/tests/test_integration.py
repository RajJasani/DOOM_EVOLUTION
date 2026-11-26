"""Integration tests for the perception → spatial memory → DoomAgent pipeline.

Verifies:
1. PerceptionBridge processes a raw observation dict end-to-end.
2. The enriched dict contains all keys expected by DoomNetwork.
3. obs["memory_features"] has the correct shape and dtype.
4. Spatial memory updates across multiple steps.
5. reset() clears all episode-scoped state.
6. Output dimensions remain consistent across many steps.
7. DoomAgent.act() returns the correct action shape (without weights).
8. The four-game-tick contract: act() is called once per decision step.
9. Edge cases: missing keys raise ValueError, empty entities work.

Run with:
    uv run pytest perception/tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.bridge import PerceptionBridge, BridgeStats, MEMORY_KEY
from perception.spatial_memory import MEMORY_FEATURE_DIM, _KEY_NAMES, _NUM_RAYS


# ---------------------------------------------------------------------------
# Realistic observation dictionary builder (matches CONTRACT.md exactly)
# ---------------------------------------------------------------------------

def _make_obs(
    health: float = 100.0,
    armor: float = 0.0,
    n_enemies: int = 0,
    enemy_dists: list | None = None,
    enemy_col1s: list | None = None,
    door_rays: list | None = None,       # list of (ray_idx, hit_type) tuples
    keys_held: Dict[str, bool] | None = None,
    ray_dists: np.ndarray | None = None,
    ray_hit_types: np.ndarray | None = None,
) -> dict:
    """Build a realistic raw observation dict matching the official env wrapper.

    All shapes and dtypes match env/doom_env/spaces.py exactly.
    """
    # --- Vitals (17,) float32 ---
    vitals = np.zeros(17, dtype=np.float32)
    vitals[0] = health
    vitals[1] = armor
    vitals[2] = 1.0   # selected_weapon
    vitals[3] = 50.0  # sel_weapon_ammo
    vitals[4] = 100.0 # bullets
    vitals[5] = 50.0  # shells
    vitals[6] = 10.0  # rockets
    vitals[7] = 100.0 # cells
    # keys [8:14]: default all 0
    if keys_held:
        key_names_ordered = list(_KEY_NAMES)
        for name, held in keys_held.items():
            if name in key_names_ordered:
                vitals[8 + key_names_ordered.index(name)] = 1.0 if held else 0.0
    vitals[14] = 5.0  # vel_forward
    vitals[15] = 0.0  # vel_right
    vitals[16] = 0.0  # vel_up

    # --- Rays ---
    if ray_dists is None:
        ray_dists = np.full(_NUM_RAYS, 0.8, dtype=np.float32)
    if ray_hit_types is None:
        ray_hit_types = np.ones(_NUM_RAYS, dtype=np.int64)   # all walls
    # Apply door overrides
    if door_rays:
        for ray_idx, ht in door_rays:
            ray_hit_types[ray_idx] = ht
            ray_dists[ray_idx] = 0.3

    rays = {
        "distance": ray_dists.astype(np.float32),
        "hit_type": ray_hit_types.astype(np.int64),
    }

    # --- Visible entities ---
    features   = np.zeros((32, 5), dtype=np.float32)
    entity_type = np.zeros(32, dtype=np.int64)
    mask        = np.zeros(32, dtype=np.int8)

    if n_enemies > 0:
        dists = enemy_dists or [1.0] * n_enemies
        col1s = enemy_col1s or [0.0] * n_enemies
        for i in range(min(n_enemies, 32)):
            mask[i] = 1
            entity_type[i] = 1   # enemy_melee
            features[i, 0] = dists[i]
            features[i, 1] = col1s[i]

    visible_entities = {
        "features":    features,
        "entity_type": entity_type,
        "mask":        mask,
    }

    # --- RGB and depth (optional, may be zeros) ---
    rgb   = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.zeros((120, 160), dtype=np.uint8)

    return {
        "vitals": vitals,
        "rays": rays,
        "visible_entities": visible_entities,
        "rgb": rgb,
        "depth": depth,
    }


# ===========================================================================
# Tests: PerceptionBridge core pipeline
# ===========================================================================

class TestPerceptionBridgeCore:
    """Verify the PerceptionBridge processes observations end-to-end."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_process_returns_dict(self):
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        assert isinstance(enriched, dict)

    def test_enriched_contains_mandatory_keys(self):
        """DoomNetwork expects vitals, rays, visible_entities at minimum."""
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        assert "vitals"            in enriched
        assert "rays"              in enriched
        assert "visible_entities"  in enriched

    def test_enriched_contains_memory_features(self):
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        assert MEMORY_KEY in enriched
        mem_vec = enriched[MEMORY_KEY]
        assert isinstance(mem_vec, np.ndarray)
        assert mem_vec.shape == (MEMORY_FEATURE_DIM,)
        assert mem_vec.dtype == np.float32

    def test_enriched_rays_keys(self):
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        assert "distance" in enriched["rays"]
        assert "hit_type" in enriched["rays"]

    def test_enriched_entity_keys(self):
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        ents = enriched["visible_entities"]
        assert "features"    in ents
        assert "entity_type" in ents
        assert "mask"        in ents

    def test_enriched_entity_mask_is_float32(self):
        """DoomNetwork's EnemyAttentionModule needs float32 mask."""
        obs = _make_obs()
        enriched = self.bridge.process(obs)
        mask = enriched["visible_entities"]["mask"]
        assert mask.dtype == np.float32

    def test_input_observation_not_mutated(self):
        """PerceptionBridge must not mutate the caller's observation dict."""
        obs = _make_obs()
        vitals_before = obs["vitals"].copy()
        _ = self.bridge.process(obs)
        np.testing.assert_array_equal(obs["vitals"], vitals_before)

    def test_memory_features_all_in_range(self):
        obs = _make_obs(n_enemies=3, enemy_dists=[0.5, 1.5, 3.0],
                        enemy_col1s=[0.0, -0.3, 0.3])
        enriched = self.bridge.process(obs)
        mem = enriched[MEMORY_KEY]
        assert float(mem.min()) >= 0.0 - 1e-6
        assert float(mem.max()) <= 1.0 + 1e-6


# ===========================================================================
# Tests: Spatial memory integration across steps
# ===========================================================================

class TestMultiStepMemory:
    """Verify memory accumulates and decays across multiple steps."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_enemy_remembered_after_disappearing(self):
        """See enemy → enemy disappears → memory still has it."""
        obs_with_enemy = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs_empty      = _make_obs()

        self.bridge.process(obs_with_enemy)
        self.bridge.process(obs_empty)   # enemy gone

        snap = self.bridge.memory.snapshot()
        assert snap.n_active_enemies == 1, "Enemy seen 1 step ago should still be remembered"

    def test_enemy_eventually_evicted(self):
        """After many steps without seeing the enemy, it should be evicted."""
        obs_with_enemy = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs_empty      = _make_obs()

        self.bridge.process(obs_with_enemy)
        for _ in range(300):
            self.bridge.process(obs_empty)

        snap = self.bridge.memory.snapshot()
        assert snap.n_active_enemies == 0

    def test_door_remembered_longer_than_enemy(self):
        """Door (static) should survive in memory longer than enemy (mobile)."""
        obs_door  = _make_obs(door_rays=[(0, 2)])   # regular door on ray 0
        obs_enemy = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs_empty = _make_obs()

        bridge_door  = PerceptionBridge()
        bridge_enemy = PerceptionBridge()

        bridge_door.process(obs_door)
        bridge_enemy.process(obs_enemy)

        for _ in range(200):
            bridge_door.process(obs_empty)
            bridge_enemy.process(obs_empty)

        door_snap  = bridge_door.memory.snapshot()
        enemy_snap = bridge_enemy.memory.snapshot()

        # Doors decay at 0.996; enemies at 0.985
        # After 200 steps: door conf ≈ 0.45, enemy conf ≈ 0.05
        assert door_snap.n_active_doors >= 1, "Door should still be remembered at step 200"
        assert enemy_snap.n_active_enemies == 0, "Enemy should be evicted by step 200"

    def test_key_tracking_across_steps(self):
        """Pick up blue key → it stays held across steps."""
        obs_no_key  = _make_obs()
        obs_with_key = _make_obs(keys_held={"blue_card": True})

        self.bridge.process(obs_no_key)
        self.bridge.process(obs_with_key)

        keys = self.bridge.memory.keys_held()
        assert keys["blue_card"] is True

    def test_step_counter_increments(self):
        for _ in range(10):
            self.bridge.process(_make_obs())
        assert self.bridge.episode_step == 10

    def test_memory_features_dimension_stable_across_steps(self):
        """The (64,) shape must hold across any number of steps."""
        for step in range(50):
            n_enemies = step % 5
            obs = _make_obs(n_enemies=n_enemies,
                            enemy_dists=[float(i + 1) * 0.5 for i in range(n_enemies)])
            enriched = self.bridge.process(obs)
            mem = enriched[MEMORY_KEY]
            assert mem.shape == (MEMORY_FEATURE_DIM,), f"Step {step}: shape mismatch"
            assert mem.dtype == np.float32, f"Step {step}: dtype mismatch"


# ===========================================================================
# Tests: reset() contract
# ===========================================================================

class TestResetContract:
    """Verify that reset() clears all episode-scoped state."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_reset_clears_memory(self):
        self.bridge.process(_make_obs(n_enemies=3,
                                      enemy_dists=[0.5, 1.0, 1.5],
                                      enemy_col1s=[0.0, -0.3, 0.3]))
        self.bridge.process(_make_obs(door_rays=[(0, 2)]))

        # Verify state before reset
        snap = self.bridge.memory.snapshot()
        assert snap.n_active_enemies > 0 or snap.n_active_doors > 0

        self.bridge.reset()

        # After reset everything is cleared
        snap = self.bridge.memory.snapshot()
        assert snap.n_active_enemies == 0
        assert snap.n_active_doors == 0
        assert snap.n_keys_held == 0
        assert snap.total_steps == 0

    def test_reset_clears_step_counter(self):
        for _ in range(5):
            self.bridge.process(_make_obs())
        assert self.bridge.episode_step == 5
        self.bridge.reset()
        assert self.bridge.episode_step == 0

    def test_reset_clears_diagnostics(self):
        self.bridge.process(_make_obs())
        assert self.bridge.last_stats is not None
        self.bridge.reset()
        assert self.bridge.last_stats is None
        assert self.bridge.last_processed is None

    def test_post_reset_process_works(self):
        """Process must work normally after reset."""
        self.bridge.process(_make_obs(n_enemies=1))
        self.bridge.reset()
        enriched = self.bridge.process(_make_obs())
        assert MEMORY_KEY in enriched
        assert enriched[MEMORY_KEY].shape == (MEMORY_FEATURE_DIM,)

    def test_multi_episode_simulation(self):
        """Simulate 3 episodes of 20 steps each."""
        for episode in range(3):
            self.bridge.reset()
            for step in range(20):
                obs = _make_obs(n_enemies=step % 3)
                enriched = self.bridge.process(obs)
                assert enriched[MEMORY_KEY].shape == (MEMORY_FEATURE_DIM,)
            assert self.bridge.episode_step == 20


# ===========================================================================
# Tests: BridgeStats diagnostics
# ===========================================================================

class TestBridgeStats:
    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_stats_populated_after_process(self):
        self.bridge.process(_make_obs())
        stats = self.bridge.last_stats
        assert stats is not None
        assert isinstance(stats, BridgeStats)

    def test_stats_step_matches_bridge_step(self):
        self.bridge.process(_make_obs())
        self.bridge.process(_make_obs())
        assert self.bridge.last_stats.step == 1  # 0-indexed step of the last process

    def test_stats_enemy_count(self):
        self.bridge.process(_make_obs(n_enemies=2, enemy_dists=[0.5, 1.5],
                                      enemy_col1s=[-0.3, 0.3]))
        assert self.bridge.last_stats.n_active_enemies == 2

    def test_stats_memory_feature_dim(self):
        self.bridge.process(_make_obs())
        assert self.bridge.last_stats.memory_feature_dim == MEMORY_FEATURE_DIM

    def test_stats_pipeline_valid(self):
        self.bridge.process(_make_obs())
        assert self.bridge.last_stats.pipeline_valid is True


# ===========================================================================
# Tests: Error handling
# ===========================================================================

class TestErrorHandling:
    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_missing_vitals_raises(self):
        obs = _make_obs()
        del obs["vitals"]
        with pytest.raises(ValueError, match="vitals"):
            self.bridge.process(obs)

    def test_missing_rays_raises(self):
        obs = _make_obs()
        del obs["rays"]
        with pytest.raises(ValueError, match="rays"):
            self.bridge.process(obs)

    def test_missing_visible_entities_raises(self):
        obs = _make_obs()
        del obs["visible_entities"]
        with pytest.raises(ValueError, match="visible_entities"):
            self.bridge.process(obs)

    def test_missing_ray_distance_raises(self):
        obs = _make_obs()
        del obs["rays"]["distance"]
        with pytest.raises(ValueError, match="distance"):
            self.bridge.process(obs)

    def test_missing_entity_mask_raises(self):
        obs = _make_obs()
        del obs["visible_entities"]["mask"]
        with pytest.raises(ValueError, match="mask"):
            self.bridge.process(obs)


# ===========================================================================
# Tests: DoomAgent integration (without trained weights)
# ===========================================================================

class TestDoomAgentIntegration:
    """Test DoomAgent with no trained weights (random fallback mode).

    These tests verify the agent lifecycle: init → reset → act.
    """

    def _make_agent(self):
        """Create a DoomAgent with mock spaces."""
        obs_space = SimpleNamespace()
        act_space = SimpleNamespace(shape=(9,))
        return _import_agent_class()(obs_space, act_space, "cpu")

    def test_agent_init(self):
        agent = self._make_agent()
        assert agent.bridge is not None
        assert agent.model is None   # no weights → None

    def test_agent_reset(self):
        agent = self._make_agent()
        agent.reset()
        assert agent.bridge.episode_step == 0
        assert agent.hx is None

    def test_agent_act_returns_correct_shape(self):
        agent = self._make_agent()
        agent.reset()
        obs = _make_obs()
        action = agent.act(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (9,)
        assert action.dtype == np.int8

    def test_agent_act_values_binary(self):
        agent = self._make_agent()
        agent.reset()
        for _ in range(10):
            action = agent.act(_make_obs())
            assert set(action.tolist()).issubset({0, 1})

    def test_agent_multi_step(self):
        """Simulate 50 decision steps (200 game ticks at FRAME_SKIP=4)."""
        agent = self._make_agent()
        agent.reset()
        for step in range(50):
            obs = _make_obs(n_enemies=step % 3)
            action = agent.act(obs)
            assert action.shape == (9,)

    def test_agent_multi_episode(self):
        """Simulate 3 episodes of 20 steps each."""
        agent = self._make_agent()
        for _ in range(3):
            agent.reset()
            for step in range(20):
                action = agent.act(_make_obs())
                assert action.shape == (9,)

    def test_agent_bridge_memory_resets_between_episodes(self):
        agent = self._make_agent()
        # Episode 1: see enemies
        agent.reset()
        agent.act(_make_obs(n_enemies=3, enemy_dists=[0.5, 1.0, 1.5],
                            enemy_col1s=[-0.3, 0.0, 0.3]))
        snap1 = agent.bridge.memory.snapshot()
        assert snap1.n_active_enemies > 0

        # Episode 2: reset should clear
        agent.reset()
        snap2 = agent.bridge.memory.snapshot()
        assert snap2.n_active_enemies == 0
        assert snap2.total_steps == 0


# ---------------------------------------------------------------------------
# Helper: import DoomAgent lazily (may fail if torch is not installed)
# ---------------------------------------------------------------------------

def _import_agent_class():
    """Import DoomAgent from submission/agent.py."""
    import importlib.util
    agent_path = Path(__file__).resolve().parents[2] / "submission" / "agent.py"
    spec = importlib.util.spec_from_file_location("submission.agent", agent_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DoomAgent
"""Additional integration tests for Sections B-H audit items.

These tests will be appended to perception/tests/test_integration.py
after the fix subagents finish.
"""

# ===========================================================================
# Tests: Entity angle extraction uses sin/cos (Bug 1 regression)
# ===========================================================================

class TestEntityAngleExtraction:
    """Verify that spatial memory correctly extracts angles from sin/cos columns."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_entity_straight_ahead(self):
        """Entity at angle 0° has sin=0, cos=1."""
        obs = _make_obs(n_enemies=1, enemy_dists=[1.0], enemy_col1s=[0.0])
        # col1=sin(0)=0, col2=cos(0)=1 → angle 0°
        # Modify features to have sin/cos encoding
        obs["visible_entities"]["features"][0, 1] = 0.0   # sin(0)
        obs["visible_entities"]["features"][0, 2] = 1.0   # cos(0)
        self.bridge.process(obs)
        enemies = self.bridge.memory.active_enemies()
        assert len(enemies) >= 1
        # Angle should be close to 0° (ahead)
        assert abs(enemies[0].angle_deg) < 5.0 or abs(enemies[0].angle_deg - 360.0) < 5.0

    def test_entity_to_the_left(self):
        """Entity at angle 90° (left) has sin=1, cos=0."""
        obs = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs["visible_entities"]["features"][0, 1] = 1.0   # sin(90°)
        obs["visible_entities"]["features"][0, 2] = 0.0   # cos(90°)
        obs["visible_entities"]["mask"][0] = 1
        obs["visible_entities"]["entity_type"][0] = 1
        self.bridge.process(obs)
        enemies = self.bridge.memory.active_enemies()
        assert len(enemies) >= 1
        assert abs(enemies[0].angle_deg - 90.0) < 5.0

    def test_entity_behind(self):
        """Entity at angle 180° has sin=0, cos=-1."""
        obs = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs["visible_entities"]["features"][0, 1] = 0.0   # sin(180°)
        obs["visible_entities"]["features"][0, 2] = -1.0  # cos(180°)
        obs["visible_entities"]["mask"][0] = 1
        obs["visible_entities"]["entity_type"][0] = 1
        self.bridge.process(obs)
        enemies = self.bridge.memory.active_enemies()
        assert len(enemies) >= 1
        assert abs(enemies[0].angle_deg - 180.0) < 5.0

    def test_entity_to_the_right(self):
        """Entity at angle 270° (right) has sin=-1, cos=0."""
        obs = _make_obs(n_enemies=1, enemy_dists=[1.0])
        obs["visible_entities"]["features"][0, 1] = -1.0  # sin(270°)
        obs["visible_entities"]["features"][0, 2] = 0.0   # cos(270°)
        obs["visible_entities"]["mask"][0] = 1
        obs["visible_entities"]["entity_type"][0] = 1
        self.bridge.process(obs)
        enemies = self.bridge.memory.active_enemies()
        assert len(enemies) >= 1
        assert abs(enemies[0].angle_deg - 270.0) < 5.0


# ===========================================================================
# Tests: Door distance not double-normalised (Bug 2 regression)
# ===========================================================================

class TestDoorDistanceScale:
    """Verify door distances use the same scale as entity distances."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_door_distance_not_quarter_scaled(self):
        """A door at ray distance 0.5 should not appear at distance 0.125 in memory."""
        obs = _make_obs(door_rays=[(0, 2)])  # door at ray 0, distance 0.3
        obs["rays"]["distance"][0] = 0.5     # override to 0.5
        self.bridge.process(obs)
        doors = self.bridge.memory.active_doors()
        assert len(doors) >= 1
        # dist_norm should be 0.5 * 4.0 = 2.0 (on the [0,4] entity scale)
        assert doors[0].dist_norm == pytest.approx(2.0, abs=0.1)

    def test_door_and_enemy_distances_comparable(self):
        """An enemy at dist=2.0 and a door at ray dist=0.5 (→ 2.0 on entity scale)
        should produce comparable distances in the feature vector."""
        obs = _make_obs(n_enemies=1, enemy_dists=[2.0], door_rays=[(16, 2)])
        obs["rays"]["distance"][16] = 0.5  # 0.5 * 4.0 = 2.0 on entity scale
        self.bridge.process(obs)
        snap = self.bridge.memory.snapshot()
        # Both should report similar nearest distances (0.5 on [0,1] output scale)
        assert abs(snap.nearest_enemy_dist - snap.nearest_door_dist) < 0.2


# ===========================================================================
# Tests: Missing optional fields handled gracefully
# ===========================================================================

class TestOptionalFields:
    """Verify bridge handles missing rgb/depth without crashing."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_missing_rgb(self):
        obs = _make_obs()
        del obs["rgb"]
        enriched = self.bridge.process(obs)
        assert MEMORY_KEY in enriched
        assert enriched[MEMORY_KEY].shape == (MEMORY_FEATURE_DIM,)

    def test_missing_depth(self):
        obs = _make_obs()
        del obs["depth"]
        enriched = self.bridge.process(obs)
        assert MEMORY_KEY in enriched

    def test_missing_both_rgb_and_depth(self):
        obs = _make_obs()
        del obs["rgb"]
        del obs["depth"]
        enriched = self.bridge.process(obs)
        assert MEMORY_KEY in enriched

    def test_zero_filled_depth(self):
        obs = _make_obs()
        obs["depth"] = np.zeros((120, 160), dtype=np.uint8)
        enriched = self.bridge.process(obs)
        assert MEMORY_KEY in enriched


# ===========================================================================
# Tests: Exactly one memory update per process() call
# ===========================================================================

class TestSingleMemoryUpdate:
    """Verify memory is updated exactly once per process() call."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_step_increments_by_one(self):
        """Each process() increments the bridge step by exactly 1."""
        for i in range(5):
            self.bridge.process(_make_obs())
            assert self.bridge.episode_step == i + 1

    def test_memory_step_matches_bridge_step(self):
        """Internal memory step should match bridge step after each call."""
        for i in range(3):
            self.bridge.process(_make_obs())
            assert self.bridge.memory.step == i + 1


# ===========================================================================
# Tests: DoomAgent fallback diagnostics (Section C)
# ===========================================================================

class TestAgentDiagnostics:
    """Verify diagnostic counters are tracked correctly."""

    def _make_agent(self):
        obs_space = SimpleNamespace()
        act_space = SimpleNamespace(shape=(9,))
        return _import_agent_class()(obs_space, act_space, "cpu")

    def test_initial_counters_zero(self):
        agent = self._make_agent()
        assert agent.n_perception_fallbacks == 0
        assert agent.n_network_fallbacks == 0
        assert agent.last_error is None
        assert agent.total_steps == 0

    def test_perception_fallback_counted(self):
        """Passing invalid obs should trigger perception fallback."""
        agent = self._make_agent()
        agent.reset()
        # Pass an obs missing mandatory keys
        bad_obs = {"rgb": np.zeros((120, 160, 3), dtype=np.uint8)}
        action = agent.act(bad_obs)
        assert action.shape == (9,)
        assert agent.n_perception_fallbacks == 1
        assert agent.last_error is not None
        assert "perception" in agent.last_error

    def test_total_steps_counted(self):
        agent = self._make_agent()
        agent.reset()
        for _ in range(5):
            agent.act(_make_obs())
        assert agent.total_steps == 5

    def test_perception_and_normal_steps_counted(self):
        """Mix of valid and invalid steps should count all."""
        agent = self._make_agent()
        agent.reset()
        agent.act(_make_obs())          # valid
        agent.act({"bad": "obs"})        # perception failure
        agent.act(_make_obs())          # valid
        assert agent.total_steps == 3
        assert agent.n_perception_fallbacks == 1


# ===========================================================================
# Tests: Network input contract preservation (Section B)
# ===========================================================================

class TestNetworkInputContract:
    """Verify the enriched dict preserves all keys DoomNetwork expects."""

    def setup_method(self):
        self.bridge = PerceptionBridge()

    def test_vitals_shape_dtype(self):
        enriched = self.bridge.process(_make_obs())
        v = enriched["vitals"]
        assert isinstance(v, np.ndarray)
        assert v.shape == (17,)
        assert v.dtype == np.float32

    def test_ray_distance_shape_dtype(self):
        enriched = self.bridge.process(_make_obs())
        d = enriched["rays"]["distance"]
        assert d.shape == (64,)
        assert d.dtype == np.float32

    def test_ray_hit_type_shape_dtype(self):
        enriched = self.bridge.process(_make_obs())
        ht = enriched["rays"]["hit_type"]
        assert ht.shape == (64,)
        assert ht.dtype == np.int64

    def test_entity_features_shape_dtype(self):
        enriched = self.bridge.process(_make_obs())
        f = enriched["visible_entities"]["features"]
        assert f.shape == (32, 5)
        assert f.dtype == np.float32

    def test_entity_type_shape_dtype(self):
        enriched = self.bridge.process(_make_obs())
        et = enriched["visible_entities"]["entity_type"]
        assert et.shape == (32,)
        assert et.dtype == np.int64

    def test_entity_mask_float32(self):
        enriched = self.bridge.process(_make_obs())
        m = enriched["visible_entities"]["mask"]
        assert m.shape == (32,)
        assert m.dtype == np.float32

    def test_memory_features_present(self):
        enriched = self.bridge.process(_make_obs())
        assert MEMORY_KEY in enriched
        assert enriched[MEMORY_KEY].shape == (MEMORY_FEATURE_DIM,)
        assert enriched[MEMORY_KEY].dtype == np.float32

    def test_no_nan_or_inf_in_output(self):
        """No NaN or Inf in any output array."""
        enriched = self.bridge.process(_make_obs(n_enemies=3,
                                                  enemy_dists=[0.5, 2.0, 3.5]))
        for key in ["vitals"]:
            assert np.all(np.isfinite(enriched[key])), f"NaN/Inf in {key}"
        assert np.all(np.isfinite(enriched["rays"]["distance"]))
        assert np.all(np.isfinite(enriched["visible_entities"]["features"]))
        assert np.all(np.isfinite(enriched[MEMORY_KEY]))


# ===========================================================================
# Tests: Circular import safety
# ===========================================================================

class TestCircularImportSafety:
    """Verify no circular import when importing perception package."""

    def test_import_perception(self):
        import importlib
        mod = importlib.import_module("perception")
        assert hasattr(mod, "PerceptionBridge")
        assert hasattr(mod, "EgocentricSpatialMemory")
        assert hasattr(mod, "SensoryPipeline")

    def test_import_bridge_directly(self):
        import importlib
        mod = importlib.import_module("perception.bridge")
        assert hasattr(mod, "PerceptionBridge")
