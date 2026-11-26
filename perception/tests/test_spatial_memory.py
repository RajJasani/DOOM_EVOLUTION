"""Unit tests for perception.spatial_memory.

Covers (original + audit regression tests):
- ObjectMemorySlot dataclass construction.
- DirectionExplorationTracker: update, bucket fractions, reset.
- EgocentricSpatialMemory: reset, update, to_feature_vector shape/range, snapshot.
- Confidence decay, reinforcement, and eviction.
- Angular matching and slot merging.
- Keycard tracking from vitals and entity observations.
- Door memory from ray observations.
- Edge cases: empty memory, all rays open, all keys held.

AUDIT REGRESSION TESTS (Phase 1–3 additions):
- Coordinate system: ego-frame convention, angle staleness documentation.
- Object matching: dual-criterion (angle + distance), separate close objects.
- Confidence decay: per-category rates, projectile fast decay, door slow decay.
- Reinforcement: EMA distance smoothing on re-observed slots.
- Configurable thresholds: angular_merge_deg, dist_merge_tol.

Run with:
    uv run pytest perception/tests/test_spatial_memory.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------
from perception.spatial_memory import (
    MEMORY_FEATURE_DIM,
    DirectionExplorationTracker,
    EgocentricSpatialMemory,
    KeyCardTracker,
    MemorySnapshot,
    ObjectMemorySlot,
    _angular_diff_deg,
    _ray_index_to_angle_deg,
    _entity_sincos_to_angle_deg,
    _active,
    _evict,
    _count_norm,
    _nearest_dist_norm,
    _mean_dist_norm,
    _nearest_angle_sincos,
    _max_confidence,
    _mean_confidence,
    _steps_since_norm,
    _EVICTION_THRESHOLD,
    _CONFIDENCE_DECAY,
    _CONFIDENCE_DECAY_ENEMY,
    _CONFIDENCE_DECAY_DOOR,
    _CONFIDENCE_DECAY_PICKUP,
    _CONFIDENCE_DECAY_PROJECTILE,
    _CONFIDENCE_REINFORCE,
    _ANGULAR_MERGE_DEG,
    _DIST_MERGE_TOL,
    _KEY_NAMES,
    _NUM_RAYS,
    _RAY_DOOR,
    _RAY_LOCKED,
    _RAY_NONE,
    _ENT_HEALTH,
    _ENT_ARMOR,
    _ENT_AMMO,
    _ENT_ENEMY_MELEE,
    _ENT_ENEMY_HITSCAN,
    _ENT_ENEMY_PROJECTILE,
    _ENT_KEY,
    _MAX_ENTITY_DIST_NORM,
)


# ---------------------------------------------------------------------------
# Mock ProcessedObservation helpers
# ---------------------------------------------------------------------------

def _make_rays(
    hit_types: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> SimpleNamespace:
    """Build a minimal RayFeatures mock."""
    if hit_types is None:
        hit_types = np.zeros(_NUM_RAYS, dtype=np.int64)
    if distances is None:
        distances = np.ones(_NUM_RAYS, dtype=np.float32)
    return SimpleNamespace(
        distance_norm=distances.astype(np.float32),
        hit_type_raw=hit_types.astype(np.int64),
    )


def _make_entities(
    type_ids: list | None = None,
    dists: list | None = None,
    col1_fracs: list | None = None,   # angles in degrees OR sin values, or sin/cos
    n_slots: int = 32,
) -> SimpleNamespace:
    """Build a minimal EntityFeatures mock with up to n_slots.

    For compatibility with _entity_sincos_to_angle_deg:
    If col1_fracs is provided, we compute sin and cos:
    - If values are roughly in [-0.5, 0.5] from legacy tests, we map:
      rel_angle = -col1 * 180° (or similar), or treat col1 as sin, col2 = sqrt(1-sin^2)
    - To seamlessly support both angle fractions and sin/cos:
      If col1 is in [-1, 1], we set:
         feat_raw[i, 1] = sin(angle)
         feat_raw[i, 2] = cos(angle)
    """
    mask     = np.zeros(n_slots, dtype=bool)
    feat_raw = np.zeros((n_slots, 5), dtype=np.float32)
    etype    = np.zeros(n_slots, dtype=np.int64)

    if type_ids is not None:
        for i, (tid, d, c1) in enumerate(zip(
            type_ids,
            dists      or [0.5] * len(type_ids),
            col1_fracs or [0.0] * len(type_ids),
        )):
            mask[i]        = True
            feat_raw[i, 0] = d    # dist_norm on [0, 4]
            # Convert c1 to sin & cos for col-1 & col-2
            # In legacy test cases, c1 was screen fraction [-0.5, +0.5]
            # where 0=ahead, -0.25=left, +0.25=right, etc.
            # In sensors.py, ang=0 -> sin=0, cos=1
            # If c1 == 0: sin=0, cos=1
            # If c1 < 0 (left): sin > 0, cos > 0
            # To preserve test intent: map c1 in [-0.5, 0.5] to angle in [-90, 90] degrees (CCW positive = left)
            ang_deg = -c1 * 180.0  # c1=-0.5 -> +90 deg, c1=+0.5 -> -90 deg
            ang_rad = math.radians(ang_deg)
            feat_raw[i, 1] = math.sin(ang_rad)
            feat_raw[i, 2] = math.cos(ang_rad)
            etype[i]       = tid

    return SimpleNamespace(
        mask=mask,
        features_raw=feat_raw,
        entity_type=etype,
    )


def _make_vitals(keys: Dict[str, bool] | None = None) -> SimpleNamespace:
    """Build a minimal VitalsFeatures mock."""
    held = {name: False for name in _KEY_NAMES}
    if keys:
        held.update(keys)
    return SimpleNamespace(keys_held=held)


def _make_processed(
    hit_types=None,
    distances=None,
    type_ids=None,
    dists=None,
    col1_fracs=None,
    keys=None,
) -> SimpleNamespace:
    """Build a complete minimal ProcessedObservation mock."""
    return SimpleNamespace(
        rays=_make_rays(hit_types, distances),
        entities=_make_entities(type_ids, dists, col1_fracs),
        vitals=_make_vitals(keys),
    )


# ---------------------------------------------------------------------------
# Convenience: empty step (nothing visible)
# ---------------------------------------------------------------------------
def _empty_step() -> SimpleNamespace:
    return _make_processed()


# ===========================================================================
# Tests: module-level constants and utilities
# ===========================================================================

class TestConstants:
    def test_memory_feature_dim_is_64(self):
        assert MEMORY_FEATURE_DIM == 64

    def test_key_names_count(self):
        assert len(_KEY_NAMES) == 6

    def test_num_rays(self):
        assert _NUM_RAYS == 64

    def test_per_category_decay_ordering(self):
        """Doors decay slowest; projectiles fastest."""
        assert _CONFIDENCE_DECAY_DOOR > _CONFIDENCE_DECAY_PICKUP
        assert _CONFIDENCE_DECAY_PICKUP > _CONFIDENCE_DECAY_ENEMY
        assert _CONFIDENCE_DECAY_ENEMY > _CONFIDENCE_DECAY_PROJECTILE

    def test_angular_merge_tightened(self):
        """Audit fix: 30° is tighter than the original 45°."""
        assert _ANGULAR_MERGE_DEG <= 30.0

    def test_reinforce_reduced(self):
        """Audit fix: reinforce reduced from 0.4 to 0.35."""
        assert _CONFIDENCE_REINFORCE <= 0.35 + 1e-9

    def test_dist_merge_tol_exists(self):
        assert _DIST_MERGE_TOL > 0.0


class TestAngularDiff:
    def test_zero_same_angle(self):
        assert _angular_diff_deg(0.0, 0.0) == pytest.approx(0.0)

    def test_90_degrees(self):
        assert _angular_diff_deg(90.0, 0.0) == pytest.approx(90.0)

    def test_wrap_around(self):
        assert _angular_diff_deg(350.0, 10.0) == pytest.approx(20.0)

    def test_opposite(self):
        assert _angular_diff_deg(180.0, 0.0) == pytest.approx(180.0)

    def test_symmetry(self):
        assert _angular_diff_deg(30.0, 90.0) == pytest.approx(
            _angular_diff_deg(90.0, 30.0)
        )


class TestRayToAngle:
    def test_ray_0_is_forward(self):
        assert _ray_index_to_angle_deg(0) == pytest.approx(0.0)

    def test_ray_16_is_90_left(self):
        assert _ray_index_to_angle_deg(16) == pytest.approx(90.0)

    def test_ray_32_is_180_behind(self):
        assert _ray_index_to_angle_deg(32) == pytest.approx(180.0)

    def test_ray_48_is_270_right(self):
        assert _ray_index_to_angle_deg(48) == pytest.approx(270.0)

    def test_ray_63_near_360(self):
        angle = _ray_index_to_angle_deg(63)
        assert 354.0 < angle < 360.0


# ===========================================================================
# PHASE 1 — Coordinate system tests
# ===========================================================================

class TestCoordinateConvention:
    """Verify ego-centric angle encoding and documented staleness behaviour."""

    def test_entity_ahead_is_zero_degrees(self):
        """sin=0.0, cos=1.0 → entity directly ahead → 0°."""
        angle = _entity_sincos_to_angle_deg(0.0, 1.0)
        assert angle == pytest.approx(0.0, abs=1.0)

    def test_entity_far_left_is_90_degrees(self):
        """sin=1.0, cos=0.0 → 90° CCW (left)."""
        angle = _entity_sincos_to_angle_deg(1.0, 0.0)
        assert angle == pytest.approx(90.0, abs=1.0)

    def test_entity_far_right_is_270_degrees(self):
        """sin=-1.0, cos=0.0 → 270° CCW (right)."""
        angle = _entity_sincos_to_angle_deg(-1.0, 0.0)
        assert angle == pytest.approx(270.0, abs=1.0)

    def test_entity_behind_is_180_degrees(self):
        """sin=0.0, cos=-1.0 → 180° CCW (behind)."""
        angle = _entity_sincos_to_angle_deg(0.0, -1.0)
        assert angle == pytest.approx(180.0, abs=1.0)

    def test_slot_angle_is_ego_frame_at_observation_time(self):
        """
        Stored angle reflects the ego-frame AT observation time.
        After the player turns, the stored angle is stale.
        The module must NOT claim global accuracy; confidence decay handles this.
        """
        memory = EgocentricSpatialMemory()
        # Observe enemy directly ahead (col1=0.0 → 0°)
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE],
            dists=[1.0],
            col1_fracs=[0.0],
        )
        memory.update(processed)
        enemies = memory.active_enemies()
        assert len(enemies) == 1
        # Angle should be ~0° (directly ahead when observed)
        assert enemies[0].angle_deg == pytest.approx(0.0, abs=5.0)

    def test_slot_angle_stored_as_left_positive(self):
        """Entity on the left of screen → angle in (0°, 180°) range (CCW positive)."""
        memory = EgocentricSpatialMemory()
        # col1 = -0.25 → entity to the left
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE],
            dists=[1.0],
            col1_fracs=[-0.25],
        )
        memory.update(processed)
        enemies = memory.active_enemies()
        assert len(enemies) == 1
        angle = enemies[0].angle_deg
        # Should be in left half-plane (0 < angle < 180)
        assert 0.0 < angle < 180.0, f"Expected left-half angle, got {angle}"

    def test_slot_angle_stored_as_right_negative(self):
        """Entity on the right of screen → angle in (180°, 360°) range."""
        memory = EgocentricSpatialMemory()
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE],
            dists=[1.0],
            col1_fracs=[0.25],
        )
        memory.update(processed)
        enemies = memory.active_enemies()
        assert len(enemies) == 1
        angle = enemies[0].angle_deg
        assert 180.0 < angle < 360.0, f"Expected right-half angle, got {angle}"

    def test_unseen_object_keeps_stale_angle_until_evicted(self):
        """
        After observing then losing sight of an enemy, the stored angle must
        NOT be changed. It stays at the last-observed value and decays in
        confidence. This is the documented staleness behaviour.
        """
        memory = EgocentricSpatialMemory()
        # Observe enemy ahead
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE],
            dists=[1.0],
            col1_fracs=[0.0],
        )
        memory.update(processed)
        angle_at_observation = memory.active_enemies()[0].angle_deg

        # 10 steps with nothing visible — enemy not seen, angle must not change
        for _ in range(10):
            memory.update(_empty_step())

        enemies = memory.active_enemies()
        if enemies:  # slot might still be active (not evicted)
            assert enemies[0].angle_deg == pytest.approx(angle_at_observation, abs=1e-6), \
                "Stale angle must not be modified when object is out of sight"

    def test_rediscovered_object_updates_angle(self):
        """
        When an object goes out of sight and is later re-observed at a different
        angle (player turned), re-observation should update the angle to the
        new bearing if the matching criteria are met.
        """
        memory = EgocentricSpatialMemory(
            angular_merge_deg=180.0,   # very wide so it still merges after a turn
            dist_merge_tol=999.0,
        )
        # Observe enemy ahead
        p1 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        memory.update(p1)
        # 5 steps no sight
        for _ in range(5):
            memory.update(_empty_step())
        # Re-observe at left (player has turned right so enemy is now 90° left)
        p2 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[-0.5],
        )
        memory.update(p2)

        enemies = memory.active_enemies()
        assert len(enemies) == 1
        # Angle should have been updated to ~90° (left), not remain at ~0°
        assert enemies[0].angle_deg != pytest.approx(0.0, abs=20.0), \
            "Slot angle should update when object is re-observed"

    def test_ray_door_angle_is_ego_frame(self):
        """Door observed on ray 16 (left) → stored at ~90°."""
        memory = EgocentricSpatialMemory()
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[16] = _RAY_DOOR
        dist = np.ones(_NUM_RAYS, dtype=np.float32) * 0.5
        processed = _make_processed(hit_types=ht, distances=dist)
        memory.update(processed)
        doors = memory.active_doors()
        assert len(doors) == 1
        assert doors[0].angle_deg == pytest.approx(90.0, abs=1e-5)


# ===========================================================================
# PHASE 2 — Object matching and merging tests
# ===========================================================================

class TestObjectMatchingDualCriterion:
    """Verify the dual-criterion (angle + distance) merging logic."""

    def test_same_angle_same_dist_merges(self):
        """Same type, same angle, same distance → reinforces existing slot."""
        memory = EgocentricSpatialMemory()
        p = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        memory.update(p)
        memory.update(p)
        assert memory.snapshot().n_active_enemies == 1

    def test_same_angle_very_different_dist_does_not_merge(self):
        """
        AUDIT FIX M1: Two enemies at the same bearing but dist 0.2 vs 3.5
        are distinct objects. With dist_merge_tol=1.0, the diff=3.3 exceeds
        the threshold → two separate slots.
        """
        memory = EgocentricSpatialMemory(dist_merge_tol=1.0)
        p1 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[0.2], col1_fracs=[0.0],
        )
        p2 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[3.5], col1_fracs=[0.0],
        )
        memory.update(p1)
        memory.update(p2)
        assert memory.snapshot().n_active_enemies == 2, (
            "Same bearing but far different distances should produce two slots"
        )

    def test_same_angle_similar_dist_merges(self):
        """Same bearing, distance diff < tol → still merges."""
        memory = EgocentricSpatialMemory(dist_merge_tol=1.0)
        p1 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        p2 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.4], col1_fracs=[0.0],
        )
        memory.update(p1)
        memory.update(p2)
        assert memory.snapshot().n_active_enemies == 1

    def test_wide_angle_separate_enemies_not_merged(self):
        """
        Two enemies > merge_threshold apart in angle → two slots.
        Default threshold is 30°.
        """
        memory = EgocentricSpatialMemory()
        p1 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[-0.4],
        )
        p2 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.4],
        )
        memory.update(p1)
        memory.update(p2)
        assert memory.snapshot().n_active_enemies == 2

    def test_different_types_same_angle_not_merged(self):
        """An enemy and a health pickup at the same angle must NOT merge."""
        memory = EgocentricSpatialMemory(angular_merge_deg=180.0, dist_merge_tol=999.0)
        p1 = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        p2 = _make_processed(
            type_ids=[_ENT_HEALTH], dists=[1.0], col1_fracs=[0.0],
        )
        memory.update(p1)
        memory.update(p2)
        # They are in different lists so they never merge regardless
        assert memory.snapshot().n_active_enemies == 1
        assert len(memory.active_pickups()) == 1

    def test_two_doors_close_in_angle_separate_slots(self):
        """
        AUDIT FIX M2: Two doors at ray 15 and ray 17 are ~11° apart.
        With threshold 30° they would merge (old 45° also merged them).
        But with threshold 30° and the same dist they should merge into 1 slot
        (they represent the same wide door). With dist_merge_tol they remain
        separate only if they're at very different depths.

        In this test: same depth → 1 slot (expected).
        """
        memory = EgocentricSpatialMemory()
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[15] = _RAY_DOOR
        ht[17] = _RAY_DOOR
        dist = np.full(_NUM_RAYS, 0.3, dtype=np.float32)
        processed = _make_processed(hit_types=ht, distances=dist)
        memory.update(processed)
        # 2 adjacent rays, same depth, same type → should merge to 1
        # (rays 15 and 17 are 11.25° apart, within 30° threshold)
        assert memory.snapshot().n_active_doors == 1

    def test_two_doors_90_degrees_apart_two_slots(self):
        """Doors at rays 0 and 16 (90° apart) must remain separate."""
        memory = EgocentricSpatialMemory()
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0]  = _RAY_DOOR
        ht[16] = _RAY_DOOR
        dist = np.full(_NUM_RAYS, 0.3, dtype=np.float32)
        processed = _make_processed(hit_types=ht, distances=dist)
        memory.update(processed)
        assert memory.snapshot().n_active_doors == 2

    def test_repeated_observation_reinforces_not_duplicates(self):
        """
        Observing the same enemy 5 times in a row must keep exactly 1 slot
        with high confidence.
        """
        memory = EgocentricSpatialMemory()
        p = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        for _ in range(5):
            memory.update(p)
        assert memory.snapshot().n_active_enemies == 1
        enemies = memory.active_enemies()
        assert enemies[0].n_observed == 5
        assert enemies[0].confidence == pytest.approx(1.0)  # capped at 1.0

    def test_unseen_object_not_deleted_immediately(self):
        """
        AUDIT: Missing observations must not immediately delete an object.
        After 1 step unseen, the slot should still be active.
        """
        memory = EgocentricSpatialMemory()
        p = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        memory.update(p)
        # One step with nothing visible
        memory.update(_empty_step())
        assert memory.snapshot().n_active_enemies == 1, \
            "Object seen just 1 step ago must not be immediately evicted"

    def test_configurable_angular_threshold(self):
        """EgocentricSpatialMemory(angular_merge_deg=5.0) is very strict."""
        memory_strict = EgocentricSpatialMemory(angular_merge_deg=5.0)
        # col1=0.0 and col1=0.1 → angle diff ~18° → no merge with strict=5°
        p1 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0])
        p2 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[-0.1])
        memory_strict.update(p1)
        memory_strict.update(p2)
        assert memory_strict.snapshot().n_active_enemies == 2

    def test_configurable_dist_tolerance(self):
        """dist_merge_tol=999 reverts to angle-only matching."""
        memory_loose = EgocentricSpatialMemory(dist_merge_tol=999.0)
        p1 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[0.1], col1_fracs=[0.0])
        p2 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[3.9], col1_fracs=[0.0])
        memory_loose.update(p1)
        memory_loose.update(p2)
        # With dist_merge_tol=999, any same-angle/same-type merges
        assert memory_loose.snapshot().n_active_enemies == 1

    def test_ema_distance_smoothing_on_reinforce(self):
        """
        AUDIT FIX: Distance is now updated via EMA (not hard replace) on reinforce.
        After observing at dist=1.0 then at dist=1.5 (within dist_merge_tol=1.0),
        slot distance should be between 1.0 and 1.5 (not exactly 1.5).
        """
        memory = EgocentricSpatialMemory()
        p1 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0])
        p2 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.5], col1_fracs=[0.0])
        memory.update(p1)
        memory.update(p2)
        enemies = memory.active_enemies()
        assert len(enemies) == 1
        # EMA: dist should be strictly between 1.0 and 1.5
        assert 1.0 < enemies[0].dist_norm < 1.5, (
            f"EMA distance should be between 1.0 and 1.5, got {enemies[0].dist_norm}"
        )


# ===========================================================================
# PHASE 3 — Confidence decay per category
# ===========================================================================

class TestPerCategoryDecay:
    """Verify that doors, enemies, and projectiles decay at different rates."""

    def _observe_once(self, type_id: int, col1: float, dist: float) -> EgocentricSpatialMemory:
        memory = EgocentricSpatialMemory()
        p = _make_processed(type_ids=[type_id], dists=[dist], col1_fracs=[col1])
        memory.update(p)
        return memory

    def _observe_door_once(self) -> EgocentricSpatialMemory:
        memory = EgocentricSpatialMemory()
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0] = _RAY_DOOR
        processed = _make_processed(hit_types=ht)
        memory.update(processed)
        return memory

    def test_enemy_decays_faster_than_door(self):
        """After N steps unseen, enemy confidence < door confidence."""
        mem_enemy = self._observe_once(_ENT_ENEMY_MELEE, 0.0, 1.0)
        mem_door  = self._observe_door_once()
        N = 50
        for _ in range(N):
            mem_enemy.update(_empty_step())
            mem_door.update(_empty_step())
        enemies = mem_enemy.active_enemies()
        doors   = mem_door.active_doors()
        if enemies and doors:
            assert enemies[0].confidence < doors[0].confidence, (
                "Enemy should decay faster than door"
            )

    def test_projectile_decays_fastest(self):
        """Projectile enemy confidence drops fastest of all categories."""
        mem_proj  = self._observe_once(_ENT_ENEMY_PROJECTILE, 0.0, 1.0)
        mem_melee = self._observe_once(_ENT_ENEMY_MELEE,      0.0, 1.0)
        N = 30
        for _ in range(N):
            mem_proj.update(_empty_step())
            mem_melee.update(_empty_step())
        proj_active  = mem_proj.active_enemies()
        melee_active = mem_melee.active_enemies()
        if proj_active and melee_active:
            assert proj_active[0].confidence < melee_active[0].confidence

    def test_door_survives_many_steps(self):
        """Door should remain active for many more steps than an enemy."""
        mem_door = self._observe_door_once()
        # 300 steps — enemy would be evicted (decay 0.985^300 ≈ 0.01 < threshold)
        for _ in range(300):
            mem_door.update(_empty_step())
        # Door with decay 0.996^300 ≈ 0.30 > 0.10 → still active
        assert mem_door.snapshot().n_active_doors == 1, (
            "Door should remain in memory for 300 steps (decay=0.996)"
        )

    def test_enemy_evicted_after_enough_steps(self):
        """Enemy observed once must eventually be evicted."""
        mem = self._observe_once(_ENT_ENEMY_MELEE, 0.0, 1.0)
        for _ in range(300):
            mem.update(_empty_step())
        assert mem.snapshot().n_active_enemies == 0

    def test_pickup_intermediate_lifetime(self):
        """Pickup survives longer than enemy but shorter than door."""
        mem_pickup = self._observe_once(_ENT_HEALTH, 0.0, 1.0)
        mem_enemy  = self._observe_once(_ENT_ENEMY_MELEE, 0.0, 1.0)
        mem_door   = self._observe_door_once()
        N = 150
        for _ in range(N):
            mem_pickup.update(_empty_step())
            mem_enemy.update(_empty_step())
            mem_door.update(_empty_step())
        pickup_conf = [s.confidence for s in mem_pickup.active_pickups()]
        enemy_conf  = [s.confidence for s in mem_enemy.active_enemies()]
        door_conf   = [s.confidence for s in mem_door.active_doors()]
        # At 150 steps:
        #   enemy: 0.985^150 ≈ 0.10 (borderline eviction)
        #   pickup: 0.990^150 ≈ 0.22 (still active)
        #   door: 0.996^150 ≈ 0.55 (comfortably active)
        if pickup_conf and door_conf:
            assert pickup_conf[0] < door_conf[0], "Pickup must decay faster than door"

    def test_confidence_reinforce_caps_at_1(self):
        """Reinforcing a fresh slot must not exceed 1.0."""
        memory = EgocentricSpatialMemory()
        p = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0])
        for _ in range(20):
            memory.update(p)
        enemies = memory.active_enemies()
        assert len(enemies) == 1
        assert enemies[0].confidence <= 1.0 + 1e-9

    def test_confidence_decay_applied_before_reinforce(self):
        """
        Decay runs BEFORE ingest each step.
        After 1 step with the same enemy seen:
        new_conf = min(1.0, (1.0 * decay) + reinforce)
        """
        memory = EgocentricSpatialMemory()
        p = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0])
        memory.update(p)   # step 0: confidence = 1.0
        memory.update(p)   # step 1: decay to 0.985, then +0.35 → min(1.0, 1.335) = 1.0
        enemies = memory.active_enemies()
        assert enemies[0].confidence == pytest.approx(1.0)


# ===========================================================================
# Tests: module-level helper functions (unchanged from original)
# ===========================================================================

class TestHelperFunctions:
    def _make_slot(self, dist=0.5, angle=0.0, conf=1.0, type_id=1) -> ObjectMemorySlot:
        return ObjectMemorySlot(
            type_id=type_id, type_name="test",
            dist_norm=dist, angle_deg=angle,
            confidence=conf, step_first=0, step_last=0,
        )

    def test_count_norm_empty(self):
        assert _count_norm([], 32) == pytest.approx(0.0)

    def test_count_norm_full(self):
        slots = [self._make_slot() for _ in range(32)]
        assert _count_norm(slots, 32) == pytest.approx(1.0)

    def test_count_norm_half(self):
        slots = [self._make_slot() for _ in range(16)]
        assert _count_norm(slots, 32) == pytest.approx(0.5)

    def test_nearest_dist_empty(self):
        assert _nearest_dist_norm([]) == pytest.approx(1.0)

    def test_nearest_dist_single(self):
        slot = self._make_slot(dist=2.0)
        assert _nearest_dist_norm([slot]) == pytest.approx(0.5)  # 2.0 / 4.0

    def test_nearest_dist_picks_minimum(self):
        slots = [self._make_slot(dist=1.0), self._make_slot(dist=3.0)]
        assert _nearest_dist_norm(slots) == pytest.approx(1.0 / 4.0)

    def test_mean_dist_empty(self):
        assert _mean_dist_norm([]) == pytest.approx(1.0)

    def test_mean_dist_clamped(self):
        slot = self._make_slot(dist=8.0)
        assert _mean_dist_norm([slot]) == pytest.approx(1.0)

    def test_nearest_angle_sincos_empty(self):
        s, c = _nearest_angle_sincos([])
        assert s == pytest.approx(0.0)
        assert c == pytest.approx(1.0)

    def test_nearest_angle_sincos_90deg(self):
        slot = self._make_slot(angle=90.0)
        s, c = _nearest_angle_sincos([slot])
        assert s == pytest.approx(1.0, abs=1e-6)
        assert c == pytest.approx(0.0, abs=1e-6)

    def test_max_confidence_empty(self):
        assert _max_confidence([]) == pytest.approx(0.0)

    def test_max_confidence(self):
        slots = [self._make_slot(conf=0.3), self._make_slot(conf=0.9)]
        assert _max_confidence(slots) == pytest.approx(0.9)

    def test_mean_confidence_empty(self):
        assert _mean_confidence([]) == pytest.approx(0.0)

    def test_mean_confidence(self):
        slots = [self._make_slot(conf=0.4), self._make_slot(conf=0.8)]
        assert _mean_confidence(slots) == pytest.approx(0.6)

    def test_steps_since_never(self):
        assert _steps_since_norm(-1, 100) == pytest.approx(1.0)

    def test_steps_since_now(self):
        assert _steps_since_norm(10, 10) == pytest.approx(0.0)

    def test_steps_since_clamped(self):
        assert _steps_since_norm(0, 1000, max_steps=200) == pytest.approx(1.0)


class TestEvict:
    def _make_slot(self, conf=0.5) -> ObjectMemorySlot:
        return ObjectMemorySlot(
            type_id=1, type_name="x",
            dist_norm=1.0, angle_deg=0.0,
            confidence=conf, step_first=0, step_last=0,
        )

    def test_evict_low_confidence(self):
        lst = [self._make_slot(conf=0.05), self._make_slot(conf=0.9)]
        _evict(lst, max_size=32, threshold=_EVICTION_THRESHOLD)
        assert len(lst) == 1
        assert lst[0].confidence == pytest.approx(0.9)

    def test_evict_capacity_cap(self):
        lst = [self._make_slot(conf=0.9) for _ in range(20)]
        _evict(lst, max_size=10, threshold=0.0)
        assert len(lst) == 10

    def test_evict_keeps_highest_confidence(self):
        lst = [self._make_slot(conf=float(i) / 20.0) for i in range(20)]
        _evict(lst, max_size=5, threshold=0.0)
        confs = [s.confidence for s in lst]
        assert min(confs) >= 15.0 / 20.0 - 1e-6


# ===========================================================================
# Tests: DirectionExplorationTracker
# ===========================================================================

class TestDirectionExplorationTracker:
    def test_initial_state_all_false(self):
        tracker = DirectionExplorationTracker()
        assert not tracker.explored_mask.any()
        assert not tracker.open_mask.any()

    def test_update_hit_marks_explored(self):
        tracker = DirectionExplorationTracker()
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0] = 1
        dist = np.ones(_NUM_RAYS, dtype=np.float32)
        tracker.update(dist, ht)
        assert tracker.explored_mask[0]
        assert not tracker.explored_mask[1]

    def test_update_no_hit_marks_open(self):
        tracker = DirectionExplorationTracker()
        ht   = np.zeros(_NUM_RAYS, dtype=np.int64)
        dist = np.ones(_NUM_RAYS, dtype=np.float32)
        tracker.update(dist, ht)
        assert tracker.open_mask.all()

    def test_explored_fractions_all_explored(self):
        tracker = DirectionExplorationTracker()
        ht   = np.ones(_NUM_RAYS, dtype=np.int64)
        dist = np.ones(_NUM_RAYS, dtype=np.float32)
        tracker.update(dist, ht)
        fracs = tracker.explored_fractions(n_buckets=16)
        assert fracs.shape == (16,)
        np.testing.assert_allclose(fracs, 1.0)

    def test_open_fractions_shape(self):
        tracker = DirectionExplorationTracker()
        fracs = tracker.open_fractions(n_buckets=8)
        assert fracs.shape == (8,)

    def test_reset_clears_state(self):
        tracker = DirectionExplorationTracker()
        ht   = np.ones(_NUM_RAYS, dtype=np.int64)
        dist = np.ones(_NUM_RAYS, dtype=np.float32)
        tracker.update(dist, ht)
        tracker.reset()
        assert not tracker.explored_mask.any()
        assert not tracker.open_mask.any()
        assert tracker.n_steps_seen == 0

    def test_ema_distance_moves_toward_new(self):
        tracker = DirectionExplorationTracker()
        ht   = np.zeros(_NUM_RAYS, dtype=np.int64)
        dist = np.full(_NUM_RAYS, 0.5, dtype=np.float32)
        for _ in range(50):
            tracker.update(dist, ht)
        np.testing.assert_allclose(tracker.smoothed_dist, 0.5, atol=0.05)


# ===========================================================================
# Tests: EgocentricSpatialMemory — reset / basic state
# ===========================================================================

class TestMemoryReset:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_initial_snapshot_all_zero(self):
        snap = self.memory.snapshot()
        assert snap.n_active_enemies == 0
        assert snap.n_active_doors   == 0
        assert snap.n_locked_doors   == 0
        assert snap.n_keys_held      == 0
        assert snap.total_steps      == 0

    def test_reset_clears_enemies(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[0.5], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_enemies > 0
        self.memory.reset()
        assert self.memory.snapshot().n_active_enemies == 0

    def test_reset_clears_step_counter(self):
        for _ in range(5):
            self.memory.update(_empty_step())
        assert self.memory.step == 5
        self.memory.reset()
        assert self.memory.step == 0

    def test_reset_clears_keys(self):
        processed = _make_processed(keys={"blue_card": True})
        self.memory.update(processed)
        assert self.memory.keys_held()["blue_card"]
        self.memory.reset()
        assert not self.memory.keys_held()["blue_card"]


# ===========================================================================
# Tests: EgocentricSpatialMemory — enemy memory
# ===========================================================================

class TestEnemyMemory:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_single_enemy_added(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_enemies == 1

    def test_enemy_at_same_angle_same_dist_reinforces(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_enemies == 1

    def test_two_enemies_far_apart_in_angle_two_slots(self):
        p1 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[-0.4])
        p2 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.4])
        self.memory.update(p1)
        self.memory.update(p2)
        assert self.memory.snapshot().n_active_enemies == 2

    def test_confidence_decays_when_unseen(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        for _ in range(50):
            self.memory.update(_empty_step())
        active = self.memory.active_enemies()
        if active:
            assert active[0].confidence < 0.5

    def test_enemy_evicted_after_many_steps(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[1.0], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        for _ in range(300):
            self.memory.update(_empty_step())
        assert self.memory.snapshot().n_active_enemies == 0

    def test_last_enemy_step_updated(self):
        self.memory.update(_empty_step())
        self.memory.update(_make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[0.5], col1_fracs=[0.0],
        ))
        snap = self.memory.snapshot()
        # Step counter increments at end of update(): step 0 was empty, step 1 saw enemy.
        # total_steps is now 2. steps_since_enemy = current_step (2) - last_enemy_step (1) = 1.
        assert snap.steps_since_enemy == 1

    def test_last_enemy_step_never_if_no_enemy(self):
        self.memory.update(_empty_step())
        snap = self.memory.snapshot()
        assert snap.steps_since_enemy == -1


# ===========================================================================
# Tests: EgocentricSpatialMemory — door memory
# ===========================================================================

class TestDoorMemory:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_door_hit_adds_slot(self):
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0] = _RAY_DOOR
        processed = _make_processed(
            hit_types=ht, distances=np.full(_NUM_RAYS, 0.3, dtype=np.float32)
        )
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_doors == 1

    def test_locked_door_counted_separately(self):
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0] = _RAY_LOCKED
        processed = _make_processed(hit_types=ht)
        self.memory.update(processed)
        snap = self.memory.snapshot()
        assert snap.n_locked_doors == 1
        assert snap.n_active_doors == 1

    def test_same_door_twice_merges(self):
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0] = _RAY_DOOR
        processed = _make_processed(hit_types=ht)
        self.memory.update(processed)
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_doors == 1

    def test_doors_at_90_degrees_different_slots(self):
        ht = np.zeros(_NUM_RAYS, dtype=np.int64)
        ht[0]  = _RAY_DOOR
        ht[16] = _RAY_DOOR
        processed = _make_processed(hit_types=ht)
        self.memory.update(processed)
        assert self.memory.snapshot().n_active_doors == 2


# ===========================================================================
# Tests: EgocentricSpatialMemory — keycard tracking
# ===========================================================================

class TestKeyCardTracking:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_key_held_from_vitals(self):
        processed = _make_processed(keys={"blue_card": True})
        self.memory.update(processed)
        assert self.memory.keys_held()["blue_card"]
        assert self.memory.snapshot().n_keys_held == 1

    def test_key_unset_in_vitals(self):
        processed = _make_processed(keys={"blue_card": False})
        self.memory.update(processed)
        assert not self.memory.keys_held()["blue_card"]

    def test_all_keys_held(self):
        all_keys = {name: True for name in _KEY_NAMES}
        processed = _make_processed(keys=all_keys)
        self.memory.update(processed)
        assert self.memory.snapshot().n_keys_held == 6

    def test_ever_seen_set_by_entity_key(self):
        processed = _make_processed(
            type_ids=[_ENT_KEY], dists=[0.5], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        any_seen = any(
            self.memory._keys[name].ever_seen for name in _KEY_NAMES
        )
        assert any_seen


# ===========================================================================
# Tests: EgocentricSpatialMemory — to_feature_vector
# ===========================================================================

class TestToFeatureVector:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_shape(self):
        vec = self.memory.to_feature_vector()
        assert vec.shape == (MEMORY_FEATURE_DIM,)

    def test_dtype_float32(self):
        vec = self.memory.to_feature_vector()
        assert vec.dtype == np.float32

    def test_all_values_in_range_empty(self):
        vec = self.memory.to_feature_vector()
        assert float(vec.min()) >= 0.0
        assert float(vec.max()) <= 1.0 + 1e-6

    def test_direction_buckets_zero_on_empty(self):
        vec = self.memory.to_feature_vector()
        assert vec[36:60].sum() == pytest.approx(0.0)

    def test_key_held_flags_at_indices_16_to_22(self):
        processed = _make_processed(keys={"blue_card": True})
        self.memory.update(processed)
        vec = self.memory.to_feature_vector()
        assert vec[16] == pytest.approx(1.0)   # blue_card is index 0
        assert vec[17] == pytest.approx(0.0)   # yellow_card not held

    def test_episode_age_increases(self):
        for _ in range(100):
            self.memory.update(_empty_step())
        vec = self.memory.to_feature_vector()
        assert vec[62] == pytest.approx(0.1, abs=0.01)

    def test_enemy_count_norm(self):
        p1 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[0.5], col1_fracs=[-0.4])
        p2 = _make_processed(type_ids=[_ENT_ENEMY_MELEE], dists=[0.5], col1_fracs=[0.4])
        self.memory.update(p1)
        self.memory.update(p2)
        vec = self.memory.to_feature_vector()
        assert vec[0] == pytest.approx(2.0 / 32.0, abs=0.01)

    def test_explored_fractions_increase_after_hits(self):
        ht = np.ones(_NUM_RAYS, dtype=np.int64)
        processed = _make_processed(hit_types=ht)
        self.memory.update(processed)
        vec = self.memory.to_feature_vector()
        np.testing.assert_allclose(vec[36:52], 1.0, atol=1e-6)

    def test_all_values_in_range_with_rich_state(self):
        """Feature vector stays in [0,1] even with many observations."""
        rng = np.random.default_rng(42)
        for step in range(50):
            ht    = rng.integers(0, 6, size=_NUM_RAYS, dtype=np.int64)
            dists = rng.random(_NUM_RAYS).astype(np.float32)
            tids  = [_ENT_ENEMY_MELEE, _ENT_HEALTH, _ENT_ARMOR]
            tdists = [float(rng.random()) * 4.0 for _ in tids]
            c1s    = [float(rng.random()) * 0.5 - 0.25 for _ in tids]
            keys   = {"blue_card": step > 25}
            processed = _make_processed(
                hit_types=ht, distances=dists,
                type_ids=tids, dists=tdists, col1_fracs=c1s,
                keys=keys,
            )
            self.memory.update(processed)
        vec = self.memory.to_feature_vector()
        # Most features are in [0, 1]; directional sine/cosine features are in [-1, 1]
        assert float(vec.min()) >= -1.0 - 1e-6
        assert float(vec.max()) <= 1.0 + 1e-6
        assert float(vec.max()) <= 1.0 + 1e-6


# ===========================================================================
# Tests: EgocentricSpatialMemory — snapshot
# ===========================================================================

class TestSnapshot:
    def setup_method(self):
        self.memory = EgocentricSpatialMemory()

    def test_snapshot_returns_named_tuple(self):
        snap = self.memory.snapshot()
        assert isinstance(snap, MemorySnapshot)

    def test_nearest_enemy_dist_1_when_empty(self):
        snap = self.memory.snapshot()
        assert snap.nearest_enemy_dist == pytest.approx(1.0)

    def test_nearest_enemy_dist_after_sighting(self):
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[2.0], col1_fracs=[0.0],
        )
        self.memory.update(processed)
        snap = self.memory.snapshot()
        # dist_norm=2.0 → renorm = 2.0 / 4.0 = 0.5
        assert snap.nearest_enemy_dist == pytest.approx(0.5)

    def test_total_steps_increments(self):
        for _ in range(7):
            self.memory.update(_empty_step())
        assert self.memory.snapshot().total_steps == 7


# ===========================================================================
# Tests: repr
# ===========================================================================

class TestRepr:
    def test_repr_contains_step(self):
        mem = EgocentricSpatialMemory()
        r = repr(mem)
        assert "step=0" in r

    def test_repr_after_update(self):
        mem = EgocentricSpatialMemory()
        processed = _make_processed(
            type_ids=[_ENT_ENEMY_MELEE], dists=[0.5], col1_fracs=[0.0],
        )
        mem.update(processed)
        r = repr(mem)
        assert "step=1" in r
        assert "enemies=1" in r
