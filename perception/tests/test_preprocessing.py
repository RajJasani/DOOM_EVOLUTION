"""Unit tests for perception/preprocessing.py.

These tests are intentionally dependency-free: they only require NumPy and
the standard library so they can run on any Python 3.13 install before
vizdoom or torch are set up.

Run with:
    python -m pytest perception/tests/test_preprocessing.py -v
    # or directly:
    python perception/tests/test_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root so perception package is importable without installing it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from perception.preprocessing import (
    ENTITY_MAX_DISTANCE_NORM,
    MAX_ENTITIES,
    NUM_RAYS,
    NUM_VITALS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    normalize_entity_features,
    normalize_vitals_numpy,
    process_observation,
    validate_observation,
)


# ---------------------------------------------------------------------------
# Fixtures: canonical valid observation
# ---------------------------------------------------------------------------

def _make_valid_obs(
    health: float = 100.0,
    n_entities: int = 3,
    render_buffers: bool = True,
) -> dict:
    """Create a minimal valid obs dict matching CONTRACT.md."""
    rgb   = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) if render_buffers else np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    depth = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),    dtype=np.uint8)

    vitals = np.zeros(NUM_VITALS, dtype=np.float32)
    vitals[0]  = health     # health
    vitals[1]  = 50.0       # armor
    vitals[2]  = 2.0        # weapon
    vitals[3]  = 25.0       # weapon ammo
    vitals[4]  = 50.0       # bullets
    vitals[8]  = 1.0        # blue card
    vitals[14] = 10.0       # vel_forward
    vitals[15] = -3.0       # vel_right

    ray_dist = np.ones(NUM_RAYS, dtype=np.float32) * 0.5
    ray_hit  = np.zeros(NUM_RAYS, dtype=np.int64)
    ray_dist[0] = 0.1   # wall straight ahead
    ray_hit[0]  = 1     # RAY_WALL

    feats = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
    etype = np.zeros(MAX_ENTITIES, dtype=np.int64)
    mask  = np.zeros(MAX_ENTITIES, dtype=np.int8)

    for i in range(min(n_entities, MAX_ENTITIES)):
        feats[i, 0] = 0.5 + i * 0.1   # distance
        feats[i, 1] = 0.0              # straight ahead
        feats[i, 2] = 1.0
        etype[i]    = 1                # enemy_melee
        mask[i]     = 1

    return {
        "rgb":   rgb,
        "depth": depth,
        "vitals": vitals,
        "rays":  {"distance": ray_dist, "hit_type": ray_hit},
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


# ===========================================================================
# SECTION 1: validate_observation
# ===========================================================================

class TestValidateObservation:

    def test_valid_obs_has_no_issues(self):
        obs = _make_valid_obs()
        issues = validate_observation(obs)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_missing_top_level_key(self):
        obs = _make_valid_obs()
        del obs["vitals"]
        issues = validate_observation(obs)
        assert any("vitals" in i for i in issues)

    def test_rgb_wrong_shape(self):
        obs = _make_valid_obs()
        obs["rgb"] = np.zeros((60, 80, 3), dtype=np.uint8)
        issues = validate_observation(obs)
        assert any("rgb" in i and "shape" in i for i in issues)

    def test_rgb_wrong_dtype(self):
        obs = _make_valid_obs()
        obs["rgb"] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32)
        issues = validate_observation(obs)
        assert any("rgb" in i and "dtype" in i for i in issues)

    def test_vitals_wrong_shape(self):
        obs = _make_valid_obs()
        obs["vitals"] = np.zeros(10, dtype=np.float32)
        issues = validate_observation(obs)
        assert any("vitals" in i and "shape" in i for i in issues)

    def test_vitals_nan_detected(self):
        obs = _make_valid_obs()
        obs["vitals"][0] = float("nan")
        issues = validate_observation(obs)
        assert any("NaN" in i for i in issues)

    def test_key_flags_must_be_binary(self):
        obs = _make_valid_obs()
        obs["vitals"][8] = 0.5   # invalid partial key flag
        issues = validate_observation(obs)
        assert any("key flag" in i.lower() or "key" in i for i in issues)

    def test_rays_distance_out_of_range(self):
        obs = _make_valid_obs()
        obs["rays"]["distance"][5] = 1.5  # > 1.0 is invalid for rays
        issues = validate_observation(obs)
        assert any("distance" in i and "[0,1]" in i for i in issues)

    def test_rays_hit_type_invalid_value(self):
        obs = _make_valid_obs()
        obs["rays"]["hit_type"][0] = 99
        issues = validate_observation(obs)
        assert any("hit_type" in i for i in issues)

    def test_entity_distance_exceeds_max(self):
        obs = _make_valid_obs()
        # Entity distance can be up to 4.0 (not 1.0!) — anything beyond is invalid
        obs["visible_entities"]["features"][0, 0] = 5.0  # > 4.0 max
        issues = validate_observation(obs)
        assert any("distance" in i.lower() and "max" in i.lower() for i in issues)

    def test_entity_distance_exactly_4_is_valid(self):
        """Entity distance of 4.0 is the theoretical max (4096 / 1024)."""
        obs = _make_valid_obs()
        obs["visible_entities"]["features"][0, 0] = 4.0
        issues = validate_observation(obs)
        # Only meaningful shape/dtype/NaN issues should remain
        blocking = [i for i in issues if "distance" in i.lower() and "max" in i.lower()]
        assert blocking == [], f"4.0 should be valid but got: {blocking}"

    def test_mask_wrong_dtype_array_accepted(self):
        """Mask stored as int64 (from MultiDiscrete) should still be valid."""
        obs = _make_valid_obs()
        obs["visible_entities"]["mask"] = obs["visible_entities"]["mask"].astype(np.int64)
        issues = validate_observation(obs)
        # This may flag a dtype warning but should not report mask values issue
        value_issues = [i for i in issues if "mask" in i and ("0 or 1" in i)]
        assert value_issues == []

    def test_render_buffers_false_all_zero_rgb_is_valid(self):
        """render_buffers=False produces all-zero rgb/depth — still valid shapes."""
        obs = _make_valid_obs(render_buffers=False)
        # The all-zero obs from render_buffers=False is shape-correct
        issues = validate_observation(obs)
        assert issues == [], f"Zero-buffer obs should be valid: {issues}"


# ===========================================================================
# SECTION 2: normalize_vitals_numpy
# ===========================================================================

class TestNormalizeVitalsNumpy:

    def test_output_shape_and_dtype(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        out = normalize_vitals_numpy(v)
        assert out.shape == (NUM_VITALS,)
        assert out.dtype == np.float32

    def test_does_not_modify_input(self):
        v = np.ones(NUM_VITALS, dtype=np.float32) * 50.0
        v_copy = v.copy()
        normalize_vitals_numpy(v)
        np.testing.assert_array_equal(v, v_copy)

    def test_health_200_normalises_to_2(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[0] = 200.0
        out = normalize_vitals_numpy(v)
        assert out[0] == pytest.approx(2.0)

    def test_health_100_normalises_to_1(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[0] = 100.0
        out = normalize_vitals_numpy(v)
        assert out[0] == pytest.approx(1.0)

    def test_weapon_7_normalises_to_1(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[2] = 7.0
        out = normalize_vitals_numpy(v)
        assert out[2] == pytest.approx(1.0)

    def test_ammo_zero_normalises_to_zero(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        out = normalize_vitals_numpy(v)
        np.testing.assert_allclose(out[3:8], 0.0)

    def test_ammo_log1p_applied(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[4] = 50.0  # ammo_bullets
        out = normalize_vitals_numpy(v)
        expected = float(np.log1p(50.0) / 4.0)
        assert out[4] == pytest.approx(expected, rel=1e-5)

    def test_key_flags_unchanged(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[8:14] = [1, 0, 1, 0, 0, 1]  # blue card, red card, red skull
        out = normalize_vitals_numpy(v)
        np.testing.assert_array_equal(out[8:14], v[8:14])

    def test_velocity_10_normalises_to_0_5(self):
        v = np.zeros(NUM_VITALS, dtype=np.float32)
        v[14] = 10.0
        out = normalize_vitals_numpy(v)
        assert out[14] == pytest.approx(0.5)

    def test_batched_input(self):
        """normalize_vitals_numpy must work for (..., 17) inputs."""
        batch = np.zeros((4, NUM_VITALS), dtype=np.float32)
        batch[:, 0] = 100.0
        out = normalize_vitals_numpy(batch)
        assert out.shape == (4, NUM_VITALS)
        np.testing.assert_allclose(out[:, 0], 1.0)

    def test_all_indices_covered(self):
        """No index should be accidentally skipped (output must differ from raw for non-zero input)."""
        v = np.ones(NUM_VITALS, dtype=np.float32) * 10.0
        v[8:14] = 0.0  # key flags: set to zero since they're only {0,1}
        out = normalize_vitals_numpy(v)
        # All non-key non-zero inputs should produce non-identity outputs
        for i in list(range(0, 8)) + list(range(14, 17)):
            assert out[i] != v[i], f"Index {i} was not normalised"


# ===========================================================================
# SECTION 3: normalize_entity_features
# ===========================================================================

class TestNormalizeEntityFeatures:

    def test_output_shape_and_dtype(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        out = normalize_entity_features(f)
        assert out.shape == (MAX_ENTITIES, 5)
        assert out.dtype == np.float32

    def test_does_not_modify_input(self):
        f = np.ones((MAX_ENTITIES, 5), dtype=np.float32)
        f_copy = f.copy()
        normalize_entity_features(f)
        np.testing.assert_array_equal(f, f_copy)

    def test_distance_4_normalises_to_1(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        f[0, 0] = 4.0   # max entity distance
        out = normalize_entity_features(f)
        assert out[0, 0] == pytest.approx(1.0)

    def test_distance_2_normalises_to_0_5(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        f[0, 0] = 2.0
        out = normalize_entity_features(f)
        assert out[0, 0] == pytest.approx(0.5)

    def test_distance_0_normalises_to_0(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        out = normalize_entity_features(f)
        assert out[0, 0] == pytest.approx(0.0)

    def test_sin_cos_angle_unchanged(self):
        """sin/cos are already in [-1,1] — must NOT be rescaled."""
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        f[0, 1] = 0.707   # sin(45°)
        f[0, 2] = 0.707   # cos(45°)
        out = normalize_entity_features(f)
        assert out[0, 1] == pytest.approx(0.707, rel=1e-4)
        assert out[0, 2] == pytest.approx(0.707, rel=1e-4)

    def test_velocity_clipped_at_20(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        f[0, 3] = 100.0   # extreme vel_forward
        f[0, 4] = -80.0   # extreme vel_right
        out = normalize_entity_features(f)
        assert out[0, 3] == pytest.approx(1.0)   # clipped to 20, /20 = 1.0
        assert out[0, 4] == pytest.approx(-1.0)  # clipped to -20, /20 = -1.0

    def test_velocity_10_normalises_to_0_5(self):
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        f[0, 3] = 10.0
        out = normalize_entity_features(f)
        assert out[0, 3] == pytest.approx(0.5)

    def test_padding_slots_remain_zero(self):
        """Zero-padded slots (mask=0) should produce all-zero outputs."""
        f = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
        out = normalize_entity_features(f)
        np.testing.assert_allclose(out, 0.0)

    def test_all_values_bounded_after_normalisation(self):
        """After normalisation, distance and velocity columns must be in [-1, 1]."""
        rng = np.random.default_rng(42)
        f = rng.uniform(-50, 50, (MAX_ENTITIES, 5)).astype(np.float32)
        f[:, 0] = rng.uniform(0, ENTITY_MAX_DISTANCE_NORM, MAX_ENTITIES).astype(np.float32)
        out = normalize_entity_features(f)
        # Distance col: [0, 1]
        assert np.all(out[:, 0] >= 0.0), "Normalised distance should be >= 0"
        assert np.all(out[:, 0] <= 1.0 + 1e-5), "Normalised distance should be <= 1"
        # Velocity cols: [-1, 1]
        assert np.all(np.abs(out[:, 3]) <= 1.0 + 1e-5)
        assert np.all(np.abs(out[:, 4]) <= 1.0 + 1e-5)

    def test_batched_input(self):
        """Must work for (..., 32, 5) batched inputs."""
        f = np.zeros((4, MAX_ENTITIES, 5), dtype=np.float32)
        f[:, 0, 0] = 2.0   # half max distance
        out = normalize_entity_features(f)
        assert out.shape == (4, MAX_ENTITIES, 5)
        np.testing.assert_allclose(out[:, 0, 0], 0.5)


# ===========================================================================
# SECTION 4: process_observation
# ===========================================================================

class TestProcessObservation:

    def test_returns_new_dict(self):
        obs = _make_valid_obs()
        enriched = process_observation(obs)
        assert enriched is not obs, "process_observation must return a new dict"

    def test_original_not_modified(self):
        obs = _make_valid_obs()
        orig_feats = obs["visible_entities"]["features"].copy()
        process_observation(obs, normalize_entities=True)
        np.testing.assert_array_equal(obs["visible_entities"]["features"], orig_feats)

    def test_all_keys_preserved(self):
        obs = _make_valid_obs()
        enriched = process_observation(obs)
        for key in obs:
            assert key in enriched, f"Key {key!r} missing from enriched obs"

    def test_entity_features_normalised_when_requested(self):
        obs = _make_valid_obs()
        # Put extreme velocity in entity 0
        obs["visible_entities"]["features"][0, 3] = 100.0
        enriched = process_observation(obs, normalize_entities=True)
        # After normalisation, velocity should be clipped to 1.0
        assert enriched["visible_entities"]["features"][0, 3] == pytest.approx(1.0)

    def test_entity_features_unchanged_when_not_requested(self):
        obs = _make_valid_obs()
        obs["visible_entities"]["features"][0, 3] = 100.0
        enriched = process_observation(obs, normalize_entities=False)
        assert enriched["visible_entities"]["features"][0, 3] == pytest.approx(100.0)

    def test_visual_encoder_called_when_provided(self):
        obs = _make_valid_obs()
        D = 64
        called = {"yes": False}

        def fake_encoder(rgb, depth, device):
            called["yes"] = True
            assert rgb.shape  == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            assert depth.shape == (SCREEN_HEIGHT, SCREEN_WIDTH)
            return np.zeros(D, dtype=np.float32)

        enriched = process_observation(obs, visual_encoder=fake_encoder)
        assert called["yes"], "visual_encoder was not called"
        assert "visual_embedding" in enriched
        assert enriched["visual_embedding"].shape == (D,)

    def test_no_visual_embedding_when_encoder_is_none(self):
        obs = _make_valid_obs()
        enriched = process_observation(obs, visual_encoder=None)
        assert "visual_embedding" not in enriched


# ===========================================================================
# SECTION 5: Edge cases and numerical stability
# ===========================================================================

class TestEdgeCases:

    def test_zero_health_obs_is_valid(self):
        """Health=0 (player dead) must still validate cleanly."""
        obs = _make_valid_obs(health=0.0)
        issues = validate_observation(obs)
        assert issues == []

    def test_no_entities_visible(self):
        obs = _make_valid_obs(n_entities=0)
        issues = validate_observation(obs)
        assert issues == []

    def test_max_entities_visible(self):
        obs = _make_valid_obs(n_entities=MAX_ENTITIES)
        issues = validate_observation(obs)
        assert issues == []

    def test_all_rays_open(self):
        obs = _make_valid_obs()
        obs["rays"]["distance"][:] = 1.0
        obs["rays"]["hit_type"][:]  = 0  # RAY_NONE
        issues = validate_observation(obs)
        assert issues == []

    def test_all_rays_wall(self):
        obs = _make_valid_obs()
        obs["rays"]["distance"][:] = 0.05  # very close walls
        obs["rays"]["hit_type"][:]  = 1    # RAY_WALL
        issues = validate_observation(obs)
        assert issues == []

    def test_normalise_vitals_no_nan(self):
        """Normalised vitals must never produce NaN or Inf."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = rng.uniform(-500, 500, NUM_VITALS).astype(np.float32)
            v[8:14] = rng.integers(0, 2, 6).astype(np.float32)
            v = np.clip(v, -1e6, 1e6)
            out = normalize_vitals_numpy(v)
            assert not np.any(np.isnan(out)), "NaN in normalised vitals"
            assert not np.any(np.isinf(out)), "Inf in normalised vitals"

    def test_normalise_entity_features_no_nan(self):
        """Normalised entity features must never produce NaN or Inf."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            f = rng.uniform(-1000, 1000, (MAX_ENTITIES, 5)).astype(np.float32)
            f[:, 0] = np.abs(f[:, 0]) % ENTITY_MAX_DISTANCE_NORM
            out = normalize_entity_features(f)
            assert not np.any(np.isnan(out)), "NaN in normalised entity features"
            assert not np.any(np.isinf(out)), "Inf in normalised entity features"


# ---------------------------------------------------------------------------
# Allow running directly: python perception/tests/test_preprocessing.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    test_classes = [
        TestValidateObservation,
        TestNormalizeVitalsNumpy,
        TestNormalizeEntityFeatures,
        TestProcessObservation,
        TestEdgeCases,
    ]
    total = passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            total += 1
            try:
                getattr(instance, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
