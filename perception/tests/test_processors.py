"""Unit tests for all five sensory processors.

Dependency-free: requires only NumPy and the standard library.
Run with:
    python -m pytest perception/tests/test_processors.py -v
    # or directly:
    python perception/tests/test_processors.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np
import pytest

# Make repo root importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from perception.processors.raycast_processor import (
    RaycastProcessor, RayFeatures,
    NUM_RAYS, NUM_HIT_TYPES,
    RAY_NONE, RAY_WALL, RAY_DOOR, RAY_LOCKED, RAY_ENEMY, RAY_ITEM,
)
from perception.processors.depth_processor import (
    DepthProcessor, DepthFeatures,
    SCREEN_HEIGHT, SCREEN_WIDTH,
)
from perception.processors.entity_processor import (
    EntityProcessor, EntityFeatures,
    MAX_ENTITIES, ENTITY_FEATURES, NUM_ENTITY_TYPES,
    ENTITY_DIST_MAX_NORM, COMBINED_TOKEN_DIM,
)
from perception.processors.vitals_processor import (
    VitalsProcessor, VitalsFeatures, NUM_VITALS,
)
from perception.processors.pipeline import (
    SensoryPipeline, ProcessedObservation,
)


# ===========================================================================
# Shared fixture builder
# ===========================================================================

def _make_obs(
    health: float = 100.0,
    n_entities: int = 3,
    render_buffers: bool = True,
    ray_hit: int = RAY_NONE,
    ray_dist: float = 1.0,
) -> dict:
    """Minimal valid obs dict matching CONTRACT.md."""
    rgb_val = 128 if render_buffers else 0
    depth_val = 200 if render_buffers else 0

    rgb   = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), rgb_val, dtype=np.uint8)
    depth = np.full((SCREEN_HEIGHT, SCREEN_WIDTH), depth_val, dtype=np.uint8)

    vitals = np.zeros(NUM_VITALS, dtype=np.float32)
    vitals[0]  = health     # health
    vitals[1]  = 50.0       # armor
    vitals[2]  = 2.0        # weapon
    vitals[3]  = 25.0       # sel_weapon_ammo
    vitals[4]  = 100.0      # bullets
    vitals[8]  = 1.0        # blue_card
    vitals[14] = 10.0       # vel_forward
    vitals[15] = -3.0       # vel_right

    distances = np.full(NUM_RAYS, ray_dist, dtype=np.float32)
    hit_types = np.full(NUM_RAYS, ray_hit, dtype=np.int64)

    feats = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
    etype = np.zeros(MAX_ENTITIES, dtype=np.int64)
    mask  = np.zeros(MAX_ENTITIES, dtype=np.int8)

    for i in range(min(n_entities, MAX_ENTITIES)):
        feats[i, 0] = 1.0 + i * 0.5   # distance in [0, 4.0] range
        feats[i, 1] = 0.0              # sin_angle
        feats[i, 2] = 1.0              # cos_angle
        feats[i, 3] = 5.0              # rel_vel_forward
        etype[i]    = 1                # enemy_melee
        mask[i]     = 1

    return {
        "rgb":   rgb,
        "depth": depth,
        "vitals": vitals,
        "rays":  {"distance": distances, "hit_type": hit_types},
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


# ===========================================================================
# SECTION 1: RaycastProcessor
# ===========================================================================

class TestRaycastProcessor:

    P = RaycastProcessor()

    def test_returns_rayfeatures_instance(self):
        obs = _make_obs()
        f = self.P.process(obs)
        assert isinstance(f, RayFeatures)

    def test_distance_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.distance_norm.shape == (NUM_RAYS,)
        assert f.distance_norm.dtype == np.float32

    def test_distance_clamped_to_0_1(self):
        obs = _make_obs()
        obs["rays"]["distance"][0] = 2.5   # above max
        obs["rays"]["distance"][1] = -0.3  # below min
        f = self.P.process(obs)
        assert float(f.distance_norm[0]) == pytest.approx(1.0)
        assert float(f.distance_norm[1]) == pytest.approx(0.0)

    def test_nan_distance_replaced_with_1(self):
        obs = _make_obs()
        obs["rays"]["distance"][5] = float("nan")
        f = self.P.process(obs)
        assert math.isfinite(float(f.distance_norm[5]))
        assert float(f.distance_norm[5]) == pytest.approx(1.0)

    def test_hit_type_raw_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.hit_type_raw.shape == (NUM_RAYS,)
        assert f.hit_type_raw.dtype == np.int64

    def test_invalid_hit_type_clamped_to_zero(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][3] = 99   # invalid
        f = self.P.process(obs)
        assert int(f.hit_type_raw[3]) == RAY_NONE

    def test_hit_type_onehot_shape(self):
        f = self.P.process(_make_obs())
        assert f.hit_type_onehot.shape == (NUM_RAYS, NUM_HIT_TYPES)
        assert f.hit_type_onehot.dtype == np.float32

    def test_hit_type_onehot_rows_sum_to_1(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][:] = RAY_WALL
        f = self.P.process(obs)
        np.testing.assert_allclose(f.hit_type_onehot.sum(axis=1), 1.0)

    def test_combined_tensor_shape(self):
        f = self.P.process(_make_obs())
        assert f.combined_tensor.shape == (NUM_RAYS, 1 + NUM_HIT_TYPES)
        assert f.combined_tensor.dtype == np.float32

    def test_combined_tensor_first_col_is_distance(self):
        obs = _make_obs(ray_dist=0.42)
        f = self.P.process(obs)
        np.testing.assert_allclose(f.combined_tensor[:, 0], 0.42)

    def test_wall_mask_correct(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][0] = RAY_WALL
        obs["rays"]["hit_type"][1] = RAY_WALL
        f = self.P.process(obs)
        assert f.wall_mask[0] and f.wall_mask[1]
        assert not f.wall_mask[2]

    def test_door_mask_includes_locked(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][5]  = RAY_DOOR
        obs["rays"]["hit_type"][10] = RAY_LOCKED
        f = self.P.process(obs)
        assert f.door_mask[5] and f.door_mask[10]

    def test_enemy_mask_correct(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][15] = RAY_ENEMY
        f = self.P.process(obs)
        assert f.enemy_mask[15]
        assert not f.enemy_mask[14]

    def test_open_mask_all_true_when_no_hits(self):
        obs = _make_obs(ray_hit=RAY_NONE)
        f = self.P.process(obs)
        assert f.open_mask.all()

    def test_summary_counts_sum_to_num_rays(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][0:4] = RAY_WALL
        obs["rays"]["hit_type"][4]   = RAY_ENEMY
        f = self.P.process(obs)
        s = f.summary
        total = s.n_wall + s.n_door + s.n_locked + s.n_enemy + s.n_item + s.n_open
        assert total == NUM_RAYS

    def test_forward_clearance_1_when_all_open(self):
        obs = _make_obs(ray_hit=RAY_NONE, ray_dist=1.0)
        f = self.P.process(obs)
        assert f.forward_clearance == pytest.approx(1.0)

    def test_forward_clearance_0_when_blocked(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 0.05   # all blocked close
        obs["rays"]["hit_type"][:] = RAY_WALL
        f = self.P.process(obs)
        assert f.forward_clearance == pytest.approx(0.0)

    def test_missing_rays_key_fallback(self):
        obs = _make_obs()
        del obs["rays"]
        f = self.P.process(obs)
        assert f.distance_norm.shape == (NUM_RAYS,)
        np.testing.assert_allclose(f.distance_norm, 1.0)

    def test_does_not_modify_obs(self):
        obs = _make_obs()
        orig = obs["rays"]["distance"].copy()
        self.P.process(obs)
        np.testing.assert_array_equal(obs["rays"]["distance"], orig)


# ===========================================================================
# SECTION 2: DepthProcessor
# ===========================================================================

class TestDepthProcessor:

    P = DepthProcessor()

    def test_returns_depthfeatures_instance(self):
        f = self.P.process(_make_obs())
        assert isinstance(f, DepthFeatures)

    def test_raw_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.raw.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)
        assert f.raw.dtype == np.uint8

    def test_normalized_shape_dtype_and_range(self):
        f = self.P.process(_make_obs())
        assert f.normalized.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)
        assert f.normalized.dtype == np.float32
        assert float(f.normalized.min()) >= 0.0
        assert float(f.normalized.max()) <= 1.0 + 1e-6

    def test_normalized_max_at_255(self):
        obs = _make_obs()
        obs["depth"][:] = 255
        f = self.P.process(obs)
        np.testing.assert_allclose(f.normalized, 1.0)

    def test_normalized_zero_at_0(self):
        obs = _make_obs()
        obs["depth"][:] = 0
        f = self.P.process(obs)
        np.testing.assert_allclose(f.normalized, 0.0)

    def test_is_zero_buffer_true_when_all_zero(self):
        obs = _make_obs(render_buffers=False)
        f = self.P.process(obs)
        assert f.is_zero_buffer is True

    def test_is_zero_buffer_false_when_nonzero(self):
        obs = _make_obs(render_buffers=True)
        f = self.P.process(obs)
        assert f.is_zero_buffer is False

    def test_horizon_strip_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.horizon_strip.shape == (SCREEN_WIDTH,)
        assert f.horizon_strip.dtype == np.float32

    def test_floor_strip_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.floor_strip.shape == (SCREEN_WIDTH,)
        assert f.floor_strip.dtype == np.float32

    def test_col_min_leq_col_mean(self):
        f = self.P.process(_make_obs())
        assert np.all(f.col_min <= f.col_mean + 1e-6)

    def test_compact_features_shape(self):
        f = self.P.process(_make_obs())
        expected = SCREEN_WIDTH * 3   # horizon + floor + col_min
        assert f.compact_features.shape == (expected,)
        assert f.compact_features.dtype == np.float32

    def test_compact_features_all_zero_when_buffer_zero(self):
        obs = _make_obs(render_buffers=False)
        f = self.P.process(obs)
        np.testing.assert_allclose(f.compact_features, 0.0)

    def test_missing_depth_key_fallback(self):
        obs = _make_obs()
        del obs["depth"]
        f = self.P.process(obs)
        assert f.is_zero_buffer is True
        assert f.normalized.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)

    def test_does_not_modify_obs(self):
        obs = _make_obs()
        orig = obs["depth"].copy()
        self.P.process(obs)
        np.testing.assert_array_equal(obs["depth"], orig)


# ===========================================================================
# SECTION 3: EntityProcessor
# ===========================================================================

class TestEntityProcessor:

    P = EntityProcessor()

    def test_returns_entityfeatures_instance(self):
        f = self.P.process(_make_obs())
        assert isinstance(f, EntityFeatures)

    def test_features_raw_shape_dtype(self):
        f = self.P.process(_make_obs())
        assert f.features_raw.shape == (MAX_ENTITIES, ENTITY_FEATURES)
        assert f.features_raw.dtype == np.float32

    def test_features_norm_shape_dtype(self):
        f = self.P.process(_make_obs())
        assert f.features_norm.shape == (MAX_ENTITIES, ENTITY_FEATURES)
        assert f.features_norm.dtype == np.float32

    def test_distance_col_normalised_to_0_1(self):
        """Entity dist ÷ 4.0 must map [0,4.0] → [0,1.0] (Audit Finding 3)."""
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 0] = 4.0   # max distance
        f = self.P.process(obs)
        assert float(f.features_norm[0, 0]) == pytest.approx(1.0)

    def test_distance_2_normalises_to_0_5(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 0] = 2.0
        f = self.P.process(obs)
        assert float(f.features_norm[0, 0]) == pytest.approx(0.5)

    def test_sin_cos_angle_unchanged(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 1] = 0.707
        obs["visible_entities"]["features"][0, 2] = 0.707
        f = self.P.process(obs)
        assert float(f.features_norm[0, 1]) == pytest.approx(0.707, rel=1e-4)
        assert float(f.features_norm[0, 2]) == pytest.approx(0.707, rel=1e-4)

    def test_velocity_clipped_at_20(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 3] = 100.0   # extreme
        obs["visible_entities"]["features"][0, 4] = -80.0
        f = self.P.process(obs)
        assert float(f.features_norm[0, 3]) == pytest.approx(1.0)
        assert float(f.features_norm[0, 4]) == pytest.approx(-1.0)

    def test_padding_slots_are_zero_in_features_norm(self):
        obs = _make_obs(n_entities=2)
        f = self.P.process(obs)
        pad_rows = f.features_norm[~f.mask]
        np.testing.assert_allclose(pad_rows, 0.0)

    def test_type_onehot_shape(self):
        f = self.P.process(_make_obs())
        assert f.type_onehot.shape == (MAX_ENTITIES, NUM_ENTITY_TYPES)
        assert f.type_onehot.dtype == np.float32

    def test_type_onehot_each_row_sums_to_1(self):
        f = self.P.process(_make_obs())
        np.testing.assert_allclose(f.type_onehot.sum(axis=1), 1.0)

    def test_combined_input_shape(self):
        f = self.P.process(_make_obs())
        assert f.combined_input.shape == (MAX_ENTITIES, COMBINED_TOKEN_DIM)
        assert f.combined_input.dtype == np.float32

    def test_n_visible_matches_mask(self):
        obs = _make_obs(n_entities=5)
        f = self.P.process(obs)
        assert f.n_visible == 5

    def test_zero_entities_n_visible_zero(self):
        obs = _make_obs(n_entities=0)
        f = self.P.process(obs)
        assert f.n_visible == 0

    def test_enemy_slice_count(self):
        obs = _make_obs(n_entities=3)   # all entity_type=1 (enemy_melee)
        f = self.P.process(obs)
        assert f.enemies.count == 3

    def test_pickup_slice_count_zero_when_only_enemies(self):
        obs = _make_obs(n_entities=3)
        f = self.P.process(obs)
        assert f.pickups.count == 0

    def test_pickup_slice_detects_health(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["entity_type"][0] = 5   # health
        f = self.P.process(obs)
        assert f.pickups.count == 1

    def test_has_incoming_projectile_true(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["entity_type"][0] = 4   # projectile
        f = self.P.process(obs)
        assert f.has_incoming_projectile is True

    def test_has_incoming_projectile_false(self):
        obs = _make_obs(n_entities=1)   # all enemy_melee (1)
        f = self.P.process(obs)
        assert f.has_incoming_projectile is False

    def test_invalid_entity_type_clamped(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["entity_type"][0] = 99   # invalid
        f = self.P.process(obs)
        assert int(f.entity_type[0]) == 0   # clamped to pad

    def test_nan_in_features_replaced(self):
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 0] = float("nan")
        f = self.P.process(obs)
        assert math.isfinite(float(f.features_norm[0, 0]))

    def test_does_not_modify_obs(self):
        obs = _make_obs(n_entities=2)
        orig = obs["visible_entities"]["features"].copy()
        self.P.process(obs)
        np.testing.assert_array_equal(obs["visible_entities"]["features"], orig)


# ===========================================================================
# SECTION 4: VitalsProcessor
# ===========================================================================

class TestVitalsProcessor:

    P = VitalsProcessor()

    def test_returns_vitalsfeatures_instance(self):
        f = self.P.process(_make_obs())
        assert isinstance(f, VitalsFeatures)

    def test_raw_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.raw.shape == (NUM_VITALS,)
        assert f.raw.dtype == np.float32

    def test_normalized_shape_and_dtype(self):
        f = self.P.process(_make_obs())
        assert f.normalized.shape == (NUM_VITALS,)
        assert f.normalized.dtype == np.float32

    def test_health_100_norm_is_1(self):
        obs = _make_obs(health=100.0)
        f = self.P.process(obs)
        assert f.health_norm == pytest.approx(1.0)

    def test_health_200_norm_is_2(self):
        obs = _make_obs(health=200.0)
        f = self.P.process(obs)
        assert f.health_norm == pytest.approx(2.0)

    def test_is_low_health_false_at_100(self):
        f = self.P.process(_make_obs(health=100.0))
        assert f.is_low_health is False

    def test_is_low_health_true_at_20(self):
        f = self.P.process(_make_obs(health=20.0))
        assert f.is_low_health is True

    def test_is_critical_false_at_100(self):
        f = self.P.process(_make_obs(health=100.0))
        assert f.is_critical is False

    def test_is_critical_true_at_5(self):
        f = self.P.process(_make_obs(health=5.0))
        assert f.is_critical is True

    def test_dead_player_health_zero_no_crash(self):
        f = self.P.process(_make_obs(health=0.0))
        assert f.health == 0.0
        assert f.is_critical is True

    def test_speed_calculation(self):
        obs = _make_obs()
        obs["vitals"][14] = 3.0   # vel_forward
        obs["vitals"][15] = 4.0   # vel_right
        f = self.P.process(obs)
        assert f.speed == pytest.approx(5.0)   # 3-4-5 triangle

    def test_is_moving_true_when_fast(self):
        obs = _make_obs()
        obs["vitals"][14] = 10.0
        f = self.P.process(obs)
        assert f.is_moving is True

    def test_is_moving_false_when_still(self):
        obs = _make_obs()
        obs["vitals"][14] = 0.0
        obs["vitals"][15] = 0.0
        f = self.P.process(obs)
        assert f.is_moving is False

    def test_ammo_counts_named_dict(self):
        f = self.P.process(_make_obs())
        assert "bullets" in f.ammo_counts
        assert "rockets" in f.ammo_counts
        assert isinstance(f.ammo_counts["bullets"], float)

    def test_ammo_norm_shape(self):
        f = self.P.process(_make_obs())
        assert f.ammo_norm.shape == (5,)
        assert f.ammo_norm.dtype == np.float32

    def test_key_flags_detected(self):
        obs = _make_obs()
        obs["vitals"][8] = 1.0    # blue_card
        obs["vitals"][10] = 1.0   # red_card
        f = self.P.process(obs)
        assert f.keys_held["blue_card"] is True
        assert f.keys_held["red_card"] is True
        assert f.keys_held["yellow_card"] is False

    def test_n_keys_counts_held_keys(self):
        obs = _make_obs()
        obs["vitals"][8:11] = 1.0   # 3 keys
        f = self.P.process(obs)
        assert f.n_keys == 3

    def test_keys_vector_shape(self):
        f = self.P.process(_make_obs())
        assert f.keys_vector.shape == (6,)

    def test_velocity_shape(self):
        f = self.P.process(_make_obs())
        assert f.velocity.shape == (3,)
        assert f.velocity_norm.shape == (3,)

    def test_velocity_norm_at_20(self):
        obs = _make_obs()
        obs["vitals"][14] = 20.0
        f = self.P.process(obs)
        assert f.velocity_norm[0] == pytest.approx(1.0)

    def test_nan_vitals_replaced_with_zero(self):
        obs = _make_obs()
        obs["vitals"][0] = float("nan")
        f = self.P.process(obs)
        assert math.isfinite(float(f.health))

    def test_normalized_no_nan_on_random_input(self):
        rng = np.random.default_rng(42)
        P = VitalsProcessor()
        for _ in range(50):
            obs = _make_obs()
            obs["vitals"] = rng.uniform(-500, 500, NUM_VITALS).astype(np.float32)
            obs["vitals"][8:14] = rng.integers(0, 2, 6).astype(np.float32)
            f = P.process(obs)
            assert not np.any(np.isnan(f.normalized)), "NaN in normalized vitals"
            assert not np.any(np.isinf(f.normalized)), "Inf in normalized vitals"

    def test_does_not_modify_obs(self):
        obs = _make_obs()
        orig = obs["vitals"].copy()
        self.P.process(obs)
        np.testing.assert_array_equal(obs["vitals"], orig)


# ===========================================================================
# SECTION 5: SensoryPipeline
# ===========================================================================

class TestSensoryPipeline:

    P = SensoryPipeline(validate=False)   # skip validation — no doom_env installed

    def test_returns_processed_observation(self):
        result = self.P.process(_make_obs())
        assert isinstance(result, ProcessedObservation)

    def test_all_sub_features_present(self):
        result = self.P.process(_make_obs())
        assert isinstance(result.rays,     RayFeatures)
        assert isinstance(result.depth,    DepthFeatures)
        assert isinstance(result.entities, EntityFeatures)
        assert isinstance(result.vitals,   VitalsFeatures)

    def test_network_input_has_all_required_keys(self):
        result = self.P.process(_make_obs())
        ni = result.network_input
        assert "vitals" in ni
        assert "rays" in ni
        assert "rays" in ni and "distance" in ni["rays"]
        assert "rays" in ni and "hit_type"  in ni["rays"]
        assert "visible_entities" in ni
        assert "features"    in ni["visible_entities"]
        assert "entity_type" in ni["visible_entities"]
        assert "mask"        in ni["visible_entities"]
        assert "rgb"   in ni
        assert "depth" in ni

    def test_network_input_vitals_is_normalised(self):
        obs = _make_obs(health=100.0)
        result = self.P.process(obs)
        # Normalised health should be 1.0, not 100.0
        assert float(result.network_input["vitals"][0]) == pytest.approx(1.0)

    def test_network_input_entity_features_normalised(self):
        """Entity distance col must be /4.0 in network_input (Finding 3)."""
        obs = _make_obs(n_entities=1)
        obs["visible_entities"]["features"][0, 0] = 4.0   # max distance
        result = self.P.process(obs)
        normed_dist = float(result.network_input["visible_entities"]["features"][0, 0])
        assert normed_dist == pytest.approx(1.0)

    def test_network_input_mask_is_float32(self):
        """EnemyAttentionModule expects float mask, not int8."""
        result = self.P.process(_make_obs())
        mask = result.network_input["visible_entities"]["mask"]
        assert mask.dtype == np.float32

    def test_network_input_ray_distance_shape(self):
        result = self.P.process(_make_obs())
        assert result.network_input["rays"]["distance"].shape == (NUM_RAYS,)

    def test_network_input_entity_features_shape(self):
        result = self.P.process(_make_obs())
        shape = result.network_input["visible_entities"]["features"].shape
        assert shape == (MAX_ENTITIES, ENTITY_FEATURES)

    def test_no_visual_embedding_when_no_encoder(self):
        result = self.P.process(_make_obs())
        assert "visual_embedding" not in result.network_input

    def test_visual_encoder_hook_called(self):
        called = {"yes": False}
        D = 64

        def fake_encoder(rgb, depth, device):
            called["yes"] = True
            assert rgb.shape   == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            assert depth.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)
            return np.zeros(D, dtype=np.float32)

        pipeline = SensoryPipeline(visual_encoder=fake_encoder, validate=False)
        result = pipeline.process(_make_obs())
        assert called["yes"], "visual_encoder was not called"
        assert "visual_embedding" in result.network_input
        assert result.network_input["visual_embedding"].shape == (D,)

    def test_does_not_modify_original_obs(self):
        obs = _make_obs()
        orig_vitals = obs["vitals"].copy()
        orig_feats  = obs["visible_entities"]["features"].copy()
        self.P.process(obs)
        np.testing.assert_array_equal(obs["vitals"], orig_vitals)
        np.testing.assert_array_equal(obs["visible_entities"]["features"], orig_feats)

    def test_pipeline_is_reusable_across_steps(self):
        obs1 = _make_obs(health=100.0, n_entities=3)
        obs2 = _make_obs(health=25.0,  n_entities=0)
        r1 = self.P.process(obs1)
        r2 = self.P.process(obs2)
        assert r1.vitals.health == pytest.approx(100.0)
        assert r2.vitals.health == pytest.approx(25.0)
        assert r1.entities.n_visible == 3
        assert r2.entities.n_visible == 0


# ===========================================================================
# SECTION 6: Edge cases across the full pipeline
# ===========================================================================

class TestEdgeCases:

    P = SensoryPipeline(validate=False)

    def test_dead_player_no_crash(self):
        obs = _make_obs(health=0.0)
        result = self.P.process(obs)
        assert result.vitals.health == 0.0
        assert result.vitals.is_critical is True

    def test_no_entities_visible(self):
        obs = _make_obs(n_entities=0)
        result = self.P.process(obs)
        assert result.entities.n_visible == 0
        assert result.entities.enemies.count == 0

    def test_max_entities_visible(self):
        obs = _make_obs(n_entities=MAX_ENTITIES)
        result = self.P.process(obs)
        assert result.entities.n_visible == MAX_ENTITIES

    def test_all_rays_blocked(self):
        obs = _make_obs(ray_hit=RAY_WALL, ray_dist=0.05)
        result = self.P.process(obs)
        assert result.rays.summary.n_wall == NUM_RAYS
        assert result.rays.forward_clearance == pytest.approx(0.0)

    def test_all_rays_open(self):
        obs = _make_obs(ray_hit=RAY_NONE, ray_dist=1.0)
        result = self.P.process(obs)
        assert result.rays.summary.n_open == NUM_RAYS
        assert result.rays.forward_clearance == pytest.approx(1.0)

    def test_render_buffers_false_depth_zero(self):
        obs = _make_obs(render_buffers=False)
        result = self.P.process(obs)
        assert result.depth.is_zero_buffer is True
        np.testing.assert_allclose(result.depth.compact_features, 0.0)

    def test_all_extreme_vitals_no_nan(self):
        obs = _make_obs()
        obs["vitals"][:] = 1e6
        obs["vitals"][8:14] = 0.0  # key flags must be binary
        result = self.P.process(obs)
        assert not np.any(np.isnan(result.vitals.normalized))

    def test_missing_all_keys_still_returns_result(self):
        """Completely empty obs dict must not crash."""
        result = self.P.process({})
        assert isinstance(result, ProcessedObservation)

    def test_network_input_always_has_required_shape(self):
        result = self.P.process({})   # worst-case empty obs
        ni = result.network_input
        assert ni["vitals"].shape == (NUM_VITALS,)
        assert ni["rays"]["distance"].shape == (NUM_RAYS,)
        assert ni["visible_entities"]["features"].shape == (MAX_ENTITIES, ENTITY_FEATURES)


# ---------------------------------------------------------------------------
# Allow running directly: python perception/tests/test_processors.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    test_classes = [
        TestRaycastProcessor,
        TestDepthProcessor,
        TestEntityProcessor,
        TestVitalsProcessor,
        TestSensoryPipeline,
        TestEdgeCases,
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
