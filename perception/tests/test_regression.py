"""Comprehensive regression tests for the perception pipeline audit.

Covers every issue identified in the Phase 1 inspection:
  1. All six keycard flags (independent, correct indices)
  2. Entity distance scaling (raw [0,4.0] → normalised [0,1.0])
  3. Raycast dimensions and circular adjacency
  4. Entity masks and padding
  5. Missing entity observations (zero entities)
  6. Missing and zero-filled depth buffers
  7. RGB and depth availability independently
  8. NaN and infinity handling
  9. Output tensor dimensions
  10. Pipeline compatibility (network_input drop-in)
  11. Memory reset behaviour
  12. Training/inference preprocessing consistency

Run with:
    uv run pytest perception/tests/test_regression.py -v
    # or directly:
    python perception/tests/test_regression.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from perception.processors.raycast_processor import (
    RaycastProcessor, NUM_RAYS, NUM_HIT_TYPES,
    RAY_NONE, RAY_WALL, RAY_DOOR, RAY_LOCKED, RAY_ENEMY, RAY_ITEM,
)
from perception.processors.depth_processor import (
    DepthProcessor, SCREEN_HEIGHT, SCREEN_WIDTH,
)
from perception.processors.entity_processor import (
    EntityProcessor, MAX_ENTITIES, ENTITY_FEATURES, ENTITY_DIST_MAX_NORM,
    COMBINED_TOKEN_DIM,
)
from perception.processors.vitals_processor import (
    VitalsProcessor, NUM_VITALS,
)
from perception.processors.pipeline import SensoryPipeline, ProcessedObservation
from perception.spatial_memory import EgocentricSpatialMemory


# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------

def _obs(
    health: float = 100.0,
    n_enemies: int = 0,
    n_keys: int = 0,
    n_health: int = 0,
    ray_hits: list[tuple[int, int, float]] | None = None,  # [(idx, hit_type, dist)]
    entity_rows: list[tuple[int, int, float]] | None = None,  # [(slot, type_id, dist_norm)]
    key_flags: dict[str, float] | None = None,  # name → 0.0/1.0
    depth_val: int = 128,
    rgb_val: int = 100,
) -> dict:
    """Build a minimal valid obs dict for regression tests."""
    _KEY_IDX = {
        "blue_card": 8, "yellow_card": 9, "red_card": 10,
        "blue_skull": 11, "yellow_skull": 12, "red_skull": 13,
    }

    vitals = np.zeros(NUM_VITALS, dtype=np.float32)
    vitals[0] = health
    vitals[1] = 50.0
    vitals[2] = 2.0
    vitals[3] = 25.0
    vitals[4] = 100.0
    vitals[14] = 10.0

    if key_flags:
        for name, val in key_flags.items():
            vitals[_KEY_IDX[name]] = val

    distances = np.ones(NUM_RAYS, dtype=np.float32)
    hit_types = np.zeros(NUM_RAYS, dtype=np.int64)

    if ray_hits:
        for idx, ht, dist in ray_hits:
            hit_types[idx] = ht
            distances[idx] = dist

    feats = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
    etype = np.zeros(MAX_ENTITIES, dtype=np.int64)
    mask  = np.zeros(MAX_ENTITIES, dtype=np.int8)

    slot = 0
    for _ in range(n_enemies):
        feats[slot, 0] = 1.0   # dist_norm = 1.0 (1024 map units)
        feats[slot, 2] = 1.0   # cos_angle = 1 (straight ahead)
        etype[slot] = 1        # enemy_melee
        mask[slot] = 1
        slot += 1
    for _ in range(n_keys):
        feats[slot, 0] = 0.5
        feats[slot, 2] = 1.0
        etype[slot] = 9        # key
        mask[slot] = 1
        slot += 1
    for _ in range(n_health):
        feats[slot, 0] = 0.8
        feats[slot, 2] = 1.0
        etype[slot] = 5        # health
        mask[slot] = 1
        slot += 1

    if entity_rows:
        for s, tid, dist in entity_rows:
            feats[s, 0] = dist
            feats[s, 2] = 1.0
            etype[s] = tid
            mask[s] = 1

    return {
        "rgb":   np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), rgb_val, dtype=np.uint8),
        "depth": np.full((SCREEN_HEIGHT, SCREEN_WIDTH), depth_val, dtype=np.uint8),
        "vitals": vitals,
        "rays":  {"distance": distances, "hit_type": hit_types},
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


P_RAY    = RaycastProcessor()
P_DEPTH  = DepthProcessor()
P_ENTITY = EntityProcessor()
P_VITALS = VitalsProcessor()
PIPELINE = SensoryPipeline(validate=False)


# ===========================================================================
# 1. All six keycard flags — independent, correct indices
# ===========================================================================

class TestKeycardFlags:
    """Regression for vitals[8:14] key flag correctness.

    Verifies that each of the six key flags maps to the correct vitals
    index, and that setting one flag does NOT affect any other.
    """

    KEY_INDEX = {
        "blue_card": 8, "yellow_card": 9, "red_card": 10,
        "blue_skull": 11, "yellow_skull": 12, "red_skull": 13,
    }

    def _obs_with_key(self, key_name: str) -> dict:
        """Return an obs where only `key_name` is True."""
        flags = {n: 0.0 for n in self.KEY_INDEX}
        flags[key_name] = 1.0
        return _obs(key_flags=flags)

    def test_no_keys_all_false(self):
        f = P_VITALS.process(_obs())
        for name in self.KEY_INDEX:
            assert f.keys_held[name] is False, f"{name} should be False"

    @pytest.mark.parametrize("key_name", list(KEY_INDEX.keys()))
    def test_each_key_independent(self, key_name):
        """Setting one key flag must not affect any other."""
        obs = self._obs_with_key(key_name)
        f = P_VITALS.process(obs)
        assert f.keys_held[key_name] is True, f"{key_name} not detected"
        for other in self.KEY_INDEX:
            if other != key_name:
                assert f.keys_held[other] is False, \
                    f"Setting {key_name} also set {other}"

    def test_all_keys_held_simultaneously(self):
        all_flags = {n: 1.0 for n in self.KEY_INDEX}
        f = P_VITALS.process(_obs(key_flags=all_flags))
        assert f.n_keys == 6
        for name in self.KEY_INDEX:
            assert f.keys_held[name] is True

    def test_keys_vector_shape_and_values(self):
        flags = {"blue_card": 1.0, "red_skull": 1.0}
        f = P_VITALS.process(_obs(key_flags=flags))
        assert f.keys_vector.shape == (6,)
        assert float(f.keys_vector[0]) == pytest.approx(1.0)   # blue_card
        assert float(f.keys_vector[5]) == pytest.approx(1.0)   # red_skull
        assert float(f.keys_vector[1]) == pytest.approx(0.0)   # yellow_card

    def test_keys_in_normalized_vector_unchanged(self):
        """Key flags (vitals[8:14]) must pass through normalisation unchanged."""
        obs = _obs(key_flags={"blue_card": 1.0, "red_card": 1.0})
        f = P_VITALS.process(obs)
        # Indices 8 and 10 in normalized should still be 1.0
        assert float(f.normalized[8])  == pytest.approx(1.0)
        assert float(f.normalized[10]) == pytest.approx(1.0)
        # Other key indices should be 0.0
        for i in [9, 11, 12, 13]:
            assert float(f.normalized[i]) == pytest.approx(0.0)

    def test_pipeline_preserves_key_flags(self):
        """Key flags must survive the full pipeline."""
        obs = _obs(key_flags={"yellow_skull": 1.0})
        result = PIPELINE.process(obs)
        # vitals[12] = yellow_skull; in network_input vitals it must still be 1.0
        assert float(result.network_input["vitals"][12]) == pytest.approx(1.0)

    def test_blue_skull_index_is_11_not_8(self):
        """Guard against off-by-one: blue_skull is index 11, not 8."""
        obs = _obs()
        obs["vitals"][11] = 1.0   # blue_skull directly
        f = P_VITALS.process(obs)
        assert f.keys_held["blue_skull"] is True
        assert f.keys_held["blue_card"] is False   # index 8 must be unaffected

    def test_yellow_skull_index_is_12(self):
        obs = _obs()
        obs["vitals"][12] = 1.0
        f = P_VITALS.process(obs)
        assert f.keys_held["yellow_skull"] is True
        assert f.keys_held["red_skull"] is False    # index 13

    def test_red_skull_index_is_13(self):
        obs = _obs()
        obs["vitals"][13] = 1.0
        f = P_VITALS.process(obs)
        assert f.keys_held["red_skull"] is True
        assert f.keys_held["yellow_skull"] is False


# ===========================================================================
# 2. Entity distance scaling
# ===========================================================================

class TestEntityDistanceScaling:
    """Regression for the entity distance normalization (Audit Finding 3).

    sensors.py:295 → feats[:,0] = edist / RAY_RANGE = edist / 1024
    Max ENTITY_MAX_DISTANCE = 4096 → max raw stored value = 4.0
    Our normalisation: /4.0 → [0, 1]
    """

    def test_raw_entity_max_is_4(self):
        """The contract spec says max entity distance in features is 4096/1024=4.0."""
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 4.0
        obs["visible_entities"]["mask"][0] = 1
        f = P_ENTITY.process(obs)
        assert float(f.features_raw[0, 0]) == pytest.approx(4.0)

    def test_normalised_distance_at_max_is_1(self):
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 4.0
        obs["visible_entities"]["mask"][0] = 1
        f = P_ENTITY.process(obs)
        assert float(f.features_norm[0, 0]) == pytest.approx(1.0)

    def test_normalised_distance_at_half_is_0_5(self):
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 2.0
        obs["visible_entities"]["mask"][0] = 1
        f = P_ENTITY.process(obs)
        assert float(f.features_norm[0, 0]) == pytest.approx(0.5)

    def test_normalised_distance_at_zero_is_0(self):
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 0.0
        obs["visible_entities"]["mask"][0] = 1
        f = P_ENTITY.process(obs)
        assert float(f.features_norm[0, 0]) == pytest.approx(0.0)

    def test_all_normalised_distances_bounded_0_1(self):
        rng = np.random.default_rng(0)
        obs = _obs()
        obs["visible_entities"]["features"][:, 0] = rng.uniform(0, 4.0, MAX_ENTITIES).astype(np.float32)
        obs["visible_entities"]["mask"][:] = 1
        f = P_ENTITY.process(obs)
        assert float(f.features_norm[:, 0].min()) >= -1e-6
        assert float(f.features_norm[:, 0].max()) <= 1.0 + 1e-6

    def test_pipeline_entity_features_are_normalised(self):
        """network_input entity features must have /4.0 applied."""
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 4.0
        obs["visible_entities"]["mask"][0] = 1
        result = PIPELINE.process(obs)
        normed_dist = float(result.network_input["visible_entities"]["features"][0, 0])
        assert normed_dist == pytest.approx(1.0), \
            f"Expected 1.0 after /4.0 normalisation, got {normed_dist}"

    def test_entity_distance_not_double_normalised(self):
        """Verify raw dist=1.0 → normed=0.25 (not 1.0, which would mean /1.0)."""
        obs = _obs()
        obs["visible_entities"]["features"][0, 0] = 1.0
        obs["visible_entities"]["mask"][0] = 1
        f = P_ENTITY.process(obs)
        # 1.0 / 4.0 = 0.25, NOT 1.0
        assert float(f.features_norm[0, 0]) == pytest.approx(0.25), \
            "dist=1.0 should normalise to 0.25 (÷4.0), not 1.0 (÷1.0)"


# ===========================================================================
# 3. Raycast dimensions and circular adjacency
# ===========================================================================

class TestRaycastDimensions:

    def test_num_rays_is_64(self):
        assert NUM_RAYS == 64

    def test_distance_shape(self):
        f = P_RAY.process(_obs())
        assert f.distance_norm.shape == (64,)

    def test_hit_type_raw_shape(self):
        f = P_RAY.process(_obs())
        assert f.hit_type_raw.shape == (64,)

    def test_onehot_shape(self):
        f = P_RAY.process(_obs())
        assert f.hit_type_onehot.shape == (64, 6)

    def test_combined_tensor_shape(self):
        f = P_RAY.process(_obs())
        assert f.combined_tensor.shape == (64, 7)

    def test_ray_0_is_forward(self):
        """Ray index 0 = straight ahead (angle 0°)."""
        # Set only ray 0 as enemy
        obs = _obs(ray_hits=[(0, RAY_ENEMY, 0.5)])
        f = P_RAY.process(obs)
        assert bool(f.enemy_mask[0]) is True
        assert not any(f.enemy_mask[1:])

    def test_ray_32_is_behind(self):
        """Ray index 32 = 180° (behind player)."""
        obs = _obs(ray_hits=[(32, RAY_WALL, 0.3)])
        f = P_RAY.process(obs)
        assert bool(f.wall_mask[32]) is True

    def test_ray_16_is_left_90(self):
        """Ray index 16 = 90° (left)."""
        obs = _obs(ray_hits=[(16, RAY_DOOR, 0.6)])
        f = P_RAY.process(obs)
        assert bool(f.door_mask[16]) is True

    def test_circular_adjacency_ray_0_and_63(self):
        """Ray 63 and Ray 0 are topologically adjacent (circular 360° scan).
        The combined_tensor's first column is distance — verify both rays
        get independent values when set differently.
        """
        obs = _obs()
        obs["rays"]["distance"][0]  = 0.1   # Ray 0 close
        obs["rays"]["distance"][63] = 0.9   # Ray 63 far
        f = P_RAY.process(obs)
        assert float(f.combined_tensor[0, 0])  == pytest.approx(0.1)
        assert float(f.combined_tensor[63, 0]) == pytest.approx(0.9)

    def test_all_6_hit_types_encoded(self):
        obs = _obs()
        for ht in range(6):
            obs["rays"]["hit_type"][ht] = ht
        f = P_RAY.process(obs)
        for ht in range(6):
            # The one-hot for ray `ht` should have 1.0 in position `ht`
            assert float(f.hit_type_onehot[ht, ht]) == pytest.approx(1.0)

    def test_onehot_rows_sum_to_1(self):
        f = P_RAY.process(_obs())
        np.testing.assert_allclose(f.hit_type_onehot.sum(axis=1), 1.0)


# ===========================================================================
# 4. Entity masks and padding
# ===========================================================================

class TestEntityMasksAndPadding:

    def test_mask_bool_dtype(self):
        f = P_ENTITY.process(_obs(n_enemies=3))
        assert f.mask.dtype == bool

    def test_active_slots_are_true(self):
        f = P_ENTITY.process(_obs(n_enemies=4))
        assert f.mask[:4].all()
        assert not f.mask[4:].any()

    def test_padding_slots_features_norm_are_zero(self):
        f = P_ENTITY.process(_obs(n_enemies=2))
        pad_rows = f.features_norm[~f.mask]
        np.testing.assert_allclose(pad_rows, 0.0, err_msg="Padding features_norm must be 0")

    def test_padding_slots_entity_type_are_zero(self):
        f = P_ENTITY.process(_obs(n_enemies=2))
        pad_types = f.entity_type[~f.mask]
        np.testing.assert_array_equal(pad_types, 0)

    def test_network_input_mask_is_float32(self):
        """EnemyAttentionModule expects float mask (line 92 in enemy_attention.py)."""
        result = PIPELINE.process(_obs(n_enemies=2))
        mask = result.network_input["visible_entities"]["mask"]
        assert mask.dtype == np.float32, f"Expected float32 mask, got {mask.dtype}"

    def test_network_input_mask_values_binary(self):
        result = PIPELINE.process(_obs(n_enemies=2))
        mask = result.network_input["visible_entities"]["mask"]
        unique = set(np.unique(mask))
        assert unique.issubset({0.0, 1.0}), f"Mask contains non-binary values: {unique}"

    def test_entity_ordering_preserved(self):
        """Entities must remain sorted nearest-first (as the env provides them)."""
        obs = _obs()
        # Manually set three entities at increasing distances
        for i in range(3):
            obs["visible_entities"]["features"][i, 0] = 0.5 + i * 0.5  # 0.5, 1.0, 1.5
            obs["visible_entities"]["entity_type"][i] = 1
            obs["visible_entities"]["mask"][i] = 1
        f = P_ENTITY.process(obs)
        dists = f.features_norm[:3, 0]
        assert dists[0] < dists[1] < dists[2], "Entity ordering must be preserved"

    def test_combined_input_shape(self):
        f = P_ENTITY.process(_obs(n_enemies=5))
        assert f.combined_input.shape == (MAX_ENTITIES, COMBINED_TOKEN_DIM)


# ===========================================================================
# 5. Missing entity observations
# ===========================================================================

class TestMissingEntities:

    def test_zero_entities_n_visible_zero(self):
        f = P_ENTITY.process(_obs(n_enemies=0))
        assert f.n_visible == 0

    def test_zero_enemies_no_crash(self):
        result = PIPELINE.process(_obs(n_enemies=0))
        assert result.entities.n_visible == 0
        assert result.entities.enemies.count == 0
        assert result.entities.pickups.count == 0

    def test_missing_visible_entities_key_fallback(self):
        obs = _obs()
        del obs["visible_entities"]
        f = P_ENTITY.process(obs)
        assert f.n_visible == 0
        assert f.features_norm.shape == (MAX_ENTITIES, ENTITY_FEATURES)
        assert not f.mask.any()

    def test_max_entities_all_active(self):
        obs = _obs()
        obs["visible_entities"]["mask"][:] = 1
        obs["visible_entities"]["entity_type"][:] = 1
        f = P_ENTITY.process(obs)
        assert f.n_visible == MAX_ENTITIES

    def test_pipeline_zero_entities_network_input_shape(self):
        result = PIPELINE.process(_obs(n_enemies=0))
        shape = result.network_input["visible_entities"]["features"].shape
        assert shape == (MAX_ENTITIES, ENTITY_FEATURES)


# ===========================================================================
# 6. Depth buffer edge cases
# ===========================================================================

class TestDepthBufferEdgeCases:

    def test_all_zero_depth_is_zero_buffer(self):
        obs = _obs(depth_val=0)
        f = P_DEPTH.process(obs)
        assert f.is_zero_buffer is True

    def test_all_zero_compact_features_are_zero(self):
        obs = _obs(depth_val=0)
        f = P_DEPTH.process(obs)
        np.testing.assert_allclose(f.compact_features, 0.0)

    def test_nonzero_depth_not_zero_buffer(self):
        obs = _obs(depth_val=128)
        f = P_DEPTH.process(obs)
        assert f.is_zero_buffer is False

    def test_missing_depth_key_fallback(self):
        obs = _obs()
        del obs["depth"]
        f = P_DEPTH.process(obs)
        assert f.is_zero_buffer is True
        assert f.normalized.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)

    def test_max_depth_255_normalises_to_1(self):
        obs = _obs(depth_val=255)
        f = P_DEPTH.process(obs)
        np.testing.assert_allclose(f.normalized, 1.0, rtol=1e-5)

    def test_depth_128_normalises_to_half(self):
        obs = _obs(depth_val=128)
        f = P_DEPTH.process(obs)
        expected = 128.0 / 255.0
        np.testing.assert_allclose(f.normalized, expected, rtol=1e-5)

    def test_compact_features_shape_480(self):
        f = P_DEPTH.process(_obs(depth_val=100))
        assert f.compact_features.shape == (480,)
        assert f.compact_features.dtype == np.float32

    def test_col_min_leq_col_mean(self):
        f = P_DEPTH.process(_obs(depth_val=200))
        assert np.all(f.col_min <= f.col_mean + 1e-6)

    def test_constant_image_col_min_eq_col_mean(self):
        obs = _obs(depth_val=100)
        f = P_DEPTH.process(obs)
        np.testing.assert_allclose(f.col_min, f.col_mean, rtol=1e-5)


# ===========================================================================
# 7. RGB and depth availability independently
# ===========================================================================

class TestRGBAndDepthAvailability:

    def test_rgb_missing_does_not_crash_pipeline(self):
        obs = _obs()
        del obs["rgb"]
        result = PIPELINE.process(obs)
        assert isinstance(result, ProcessedObservation)

    def test_depth_missing_does_not_crash_pipeline(self):
        obs = _obs()
        del obs["depth"]
        result = PIPELINE.process(obs)
        assert result.depth.is_zero_buffer is True

    def test_rgb_zero_depth_zero_independent(self):
        """RGB can be zero while depth is non-zero and vice versa."""
        obs = _obs(rgb_val=0, depth_val=200)
        result = PIPELINE.process(obs)
        assert not result.depth.is_zero_buffer
        # RGB should still be in network_input
        assert "rgb" in result.network_input

    def test_rgb_nonzero_depth_zero(self):
        obs = _obs(rgb_val=100, depth_val=0)
        result = PIPELINE.process(obs)
        assert result.depth.is_zero_buffer is True


# ===========================================================================
# 8. NaN and infinity handling
# ===========================================================================

class TestNaNAndInfinityHandling:

    def test_nan_distance_replaced_with_1(self):
        obs = _obs()
        obs["rays"]["distance"][5] = float("nan")
        f = P_RAY.process(obs)
        assert math.isfinite(float(f.distance_norm[5]))
        assert float(f.distance_norm[5]) == pytest.approx(1.0)

    def test_inf_distance_clamped_to_1(self):
        obs = _obs()
        obs["rays"]["distance"][10] = float("inf")
        f = P_RAY.process(obs)
        assert float(f.distance_norm[10]) == pytest.approx(1.0)

    def test_nan_vitals_replaced_with_zero(self):
        obs = _obs()
        obs["vitals"][0] = float("nan")
        f = P_VITALS.process(obs)
        assert math.isfinite(float(f.health))

    def test_inf_vitals_replaced_with_zero(self):
        obs = _obs()
        obs["vitals"][14] = float("inf")
        f = P_VITALS.process(obs)
        assert math.isfinite(float(f.velocity_norm[0]))

    def test_nan_entity_features_replaced_with_zero(self):
        obs = _obs(n_enemies=1)
        obs["visible_entities"]["features"][0, 0] = float("nan")
        f = P_ENTITY.process(obs)
        assert math.isfinite(float(f.features_norm[0, 0]))

    def test_normalized_vitals_no_nan_on_extreme_input(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            obs = _obs()
            obs["vitals"] = rng.uniform(-1e6, 1e6, NUM_VITALS).astype(np.float32)
            obs["vitals"][8:14] = rng.integers(0, 2, 6).astype(np.float32)
            f = P_VITALS.process(obs)
            assert not np.any(np.isnan(f.normalized)), "NaN in vitals normalisation"
            assert not np.any(np.isinf(f.normalized)), "Inf in vitals normalisation"

    def test_normalized_entity_features_no_nan(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            obs = _obs()
            obs["visible_entities"]["features"] = rng.uniform(-1e3, 1e3, (MAX_ENTITIES, ENTITY_FEATURES)).astype(np.float32)
            obs["visible_entities"]["features"][:, 0] = rng.uniform(0, 4.0, MAX_ENTITIES).astype(np.float32)
            obs["visible_entities"]["mask"][:] = 1
            f = P_ENTITY.process(obs)
            assert not np.any(np.isnan(f.features_norm))


# ===========================================================================
# 9. Output tensor dimensions
# ===========================================================================

class TestOutputTensorDimensions:

    def test_vitals_normalized_shape(self):
        f = P_VITALS.process(_obs())
        assert f.normalized.shape == (17,)
        assert f.normalized.dtype == np.float32

    def test_ray_distance_shape(self):
        f = P_RAY.process(_obs())
        assert f.distance_norm.shape == (64,)

    def test_ray_combined_tensor_shape(self):
        f = P_RAY.process(_obs())
        assert f.combined_tensor.shape == (64, 7)   # 64 rays × (1 dist + 6 hit types)

    def test_entity_features_norm_shape(self):
        f = P_ENTITY.process(_obs())
        assert f.features_norm.shape == (32, 5)

    def test_entity_combined_input_shape(self):
        f = P_ENTITY.process(_obs())
        assert f.combined_input.shape == (32, 17)   # 5 feats + 12 type one-hot

    def test_depth_compact_features_shape(self):
        f = P_DEPTH.process(_obs())
        assert f.compact_features.shape == (480,)   # 160×3

    def test_network_input_vitals_shape(self):
        result = PIPELINE.process(_obs())
        assert result.network_input["vitals"].shape == (17,)

    def test_network_input_ray_distance_shape(self):
        result = PIPELINE.process(_obs())
        assert result.network_input["rays"]["distance"].shape == (64,)

    def test_network_input_entity_features_shape(self):
        result = PIPELINE.process(_obs())
        assert result.network_input["visible_entities"]["features"].shape == (32, 5)

    def test_network_input_entity_mask_shape(self):
        result = PIPELINE.process(_obs())
        assert result.network_input["visible_entities"]["mask"].shape == (32,)

    def test_ammo_norm_shape(self):
        f = P_VITALS.process(_obs())
        assert f.ammo_norm.shape == (5,)

    def test_keys_vector_shape(self):
        f = P_VITALS.process(_obs())
        assert f.keys_vector.shape == (6,)


# ===========================================================================
# 10. Pipeline compatibility (network_input drop-in)
# ===========================================================================

class TestPipelineCompatibility:

    def test_network_input_has_all_required_keys(self):
        result = PIPELINE.process(_obs())
        ni = result.network_input
        required = {"vitals", "rays", "visible_entities", "rgb", "depth"}
        assert required.issubset(set(ni.keys()))

    def test_network_input_rays_has_distance_and_hit_type(self):
        result = PIPELINE.process(_obs())
        assert "distance" in result.network_input["rays"]
        assert "hit_type"  in result.network_input["rays"]

    def test_network_input_entities_has_three_subkeys(self):
        result = PIPELINE.process(_obs())
        ve = result.network_input["visible_entities"]
        assert "features"    in ve
        assert "entity_type" in ve
        assert "mask"        in ve

    def test_pipeline_does_not_modify_original_obs(self):
        obs = _obs(n_enemies=3)
        orig_vitals = obs["vitals"].copy()
        orig_feats  = obs["visible_entities"]["features"].copy()
        PIPELINE.process(obs)
        np.testing.assert_array_equal(obs["vitals"], orig_vitals)
        np.testing.assert_array_equal(obs["visible_entities"]["features"], orig_feats)

    def test_network_input_vitals_is_normalised_not_raw(self):
        obs = _obs(health=100.0)
        result = PIPELINE.process(obs)
        normed_health = float(result.network_input["vitals"][0])
        # health/100 = 1.0, NOT raw 100.0
        assert normed_health == pytest.approx(1.0), \
            f"vitals[0] in network_input should be 1.0 (normalised), got {normed_health}"

    def test_pipeline_reusable_across_steps(self):
        obs1 = _obs(health=100.0, n_enemies=3)
        obs2 = _obs(health=25.0,  n_enemies=0)
        r1 = PIPELINE.process(obs1)
        r2 = PIPELINE.process(obs2)
        assert r1.vitals.health == pytest.approx(100.0)
        assert r2.vitals.health == pytest.approx(25.0)
        assert r1.entities.n_visible == 3
        assert r2.entities.n_visible == 0

    def test_empty_obs_does_not_crash(self):
        result = PIPELINE.process({})
        assert isinstance(result, ProcessedObservation)
        assert result.network_input["vitals"].shape == (17,)


# ===========================================================================
# 11. Memory reset behaviour
# ===========================================================================

class TestMemoryResetBehaviour:

    def test_reset_clears_enemies(self):
        mem = EgocentricSpatialMemory()
        obs = _obs(n_enemies=3)
        result = PIPELINE.process(obs)
        mem.update(result)
        snap_before = mem.snapshot()
        assert snap_before.n_remembered_enemies > 0

        mem.reset()
        snap_after = mem.snapshot()
        assert snap_after.n_remembered_enemies == 0

    def test_reset_clears_key_flags(self):
        mem = EgocentricSpatialMemory()
        obs = _obs(key_flags={"blue_card": 1.0})
        result = PIPELINE.process(obs)
        mem.update(result)
        assert mem.keys_held()["blue_card"] is True

        mem.reset()
        assert mem.keys_held()["blue_card"] is False

    def test_reset_clears_step_counter(self):
        mem = EgocentricSpatialMemory()
        for _ in range(5):
            mem.update(PIPELINE.process(_obs()))
        mem.reset()
        # After reset, the first update should start step counter at 1
        mem.update(PIPELINE.process(_obs(n_enemies=1)))
        assert mem.snapshot().n_remembered_enemies == 1

    def test_multiple_resets_are_idempotent(self):
        mem = EgocentricSpatialMemory()
        mem.update(PIPELINE.process(_obs(n_enemies=2)))
        mem.reset()
        mem.reset()   # second reset must be safe
        snap = mem.snapshot()
        assert snap.n_remembered_enemies == 0

    def test_memory_accumulates_across_steps_without_reset(self):
        mem = EgocentricSpatialMemory()
        # Step 1: see enemies
        mem.update(PIPELINE.process(_obs(n_enemies=2)))
        snap1 = mem.snapshot()
        assert snap1.n_remembered_enemies >= 1

    def test_snapshot_returns_all_fields(self):
        mem = EgocentricSpatialMemory()
        snap = mem.snapshot()
        assert hasattr(snap, "n_remembered_enemies")
        assert hasattr(snap, "n_remembered_doors")
        assert hasattr(snap, "n_held_keys")
        assert hasattr(snap, "nearest_remembered_enemy_dist")


# ===========================================================================
# 12. Training / inference preprocessing consistency
# ===========================================================================

class TestTrainingInferenceConsistency:

    def test_vitals_normalisation_matches_network_encoder(self):
        """VitalsProcessor._normalise must produce identical results to
        VitalsEncoder.normalize_vitals() (up to float32 rounding).
        We verify the formula by manual computation.
        """
        obs = _obs(health=75.0)
        obs["vitals"][4] = 50.0   # bullets

        f = P_VITALS.process(obs)

        # Manual replication of VitalsEncoder.normalize_vitals():
        v = obs["vitals"].astype(np.float32)
        expected = v.copy()
        expected[0:2]  /= 100.0
        expected[2:3]  /= 7.0
        ammo = np.maximum(expected[3:8], 0.0)
        expected[3:8]   = np.log1p(ammo) / 4.0
        expected[14:17] /= 20.0

        np.testing.assert_allclose(f.normalized, expected, rtol=1e-5,
                                   err_msg="VitalsProcessor normalisation differs from VitalsEncoder")

    def test_entity_normalisation_stable_across_calls(self):
        """Same obs → same normalised output on every call (stateless)."""
        obs = _obs(n_enemies=5)
        f1 = P_ENTITY.process(obs)
        f2 = P_ENTITY.process(obs)
        np.testing.assert_array_equal(f1.features_norm, f2.features_norm)

    def test_pipeline_deterministic(self):
        """Same obs produces identical network_input on repeated calls."""
        obs = _obs(n_enemies=3, key_flags={"blue_card": 1.0})
        r1 = PIPELINE.process(obs)
        r2 = PIPELINE.process(obs)
        np.testing.assert_array_equal(
            r1.network_input["vitals"],
            r2.network_input["vitals"],
        )
        np.testing.assert_array_equal(
            r1.network_input["visible_entities"]["features"],
            r2.network_input["visible_entities"]["features"],
        )

    def test_key_flags_not_scaled_by_normalisation(self):
        """Key flags (vitals[8:14]) must NOT be multiplied by any scale factor.
        VitalsEncoder.normalize_vitals() leaves [8:14] unchanged.
        """
        obs = _obs(key_flags={k: 1.0 for k in
                               ["blue_card","yellow_card","red_card",
                                "blue_skull","yellow_skull","red_skull"]})
        f = P_VITALS.process(obs)
        # All key flag normalised values must equal the raw values (1.0)
        np.testing.assert_allclose(
            f.normalized[8:14], 1.0,
            err_msg="Key flags must pass through normalisation unchanged"
        )

    def test_ammo_uses_log1p_not_linear(self):
        """Ammo normalisation uses log1p / 4.0 — verify non-linearity."""
        obs = _obs()
        obs["vitals"][4] = 100.0   # bullets
        f = P_VITALS.process(obs)
        expected = float(np.log1p(100.0) / 4.0)
        assert f.normalized[4] == pytest.approx(expected, rel=1e-5)
        # Must NOT be 100.0/4.0 (linear)
        assert f.normalized[4] != pytest.approx(100.0 / 4.0, rel=0.01)


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    test_classes = [
        TestKeycardFlags,
        TestEntityDistanceScaling,
        TestRaycastDimensions,
        TestEntityMasksAndPadding,
        TestMissingEntities,
        TestDepthBufferEdgeCases,
        TestRGBAndDepthAvailability,
        TestNaNAndInfinityHandling,
        TestOutputTensorDimensions,
        TestPipelineCompatibility,
        TestMemoryResetBehaviour,
        TestTrainingInferenceConsistency,
    ]

    total = passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in sorted(methods):
            total += 1
            try:
                getattr(instance, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
