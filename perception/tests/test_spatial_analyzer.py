"""Unit tests for perception/spatial_analyzer.py.

Dependency-free: requires only NumPy and the standard library.
Run with:
    python -m pytest perception/tests/test_spatial_analyzer.py -v
    # or directly:
    python perception/tests/test_spatial_analyzer.py
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

from perception.spatial_analyzer import (
    SpatialAnalyzer,
    RaySummary,
    ThreatReport,
    PickupReport,
    DoorReport,
    NavigationHints,
    NUM_RAYS,
    RAY_RANGE,
    ENTITY_DIST_MAX_NORM,
    _RAY_HIT_NONE,
    _RAY_HIT_WALL,
    _RAY_HIT_DOOR,
    _RAY_HIT_LOCKED,
    _RAY_HIT_ENEMY,
    _RAY_HIT_ITEM,
    _ENEMY_TYPE_IDS,
    _KEY_TYPE_ID,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_obs(
    *,
    n_enemies: int = 0,
    n_keys: int = 0,
    n_health: int = 0,
    n_projectiles: int = 0,
    ray_hit_types: list[int] | None = None,
    ray_distances: list[float] | None = None,
    health: float = 100.0,
    has_blue_card: bool = False,
) -> dict:
    """Create a minimal obs dict for spatial analysis tests."""
    # --- rays ---
    distances = np.ones(NUM_RAYS, dtype=np.float32)
    hit_types  = np.zeros(NUM_RAYS, dtype=np.int64)   # all open by default

    if ray_hit_types is not None:
        for i, h in enumerate(ray_hit_types):
            hit_types[i] = h
    if ray_distances is not None:
        for i, d in enumerate(ray_distances):
            distances[i] = d

    # --- entities ---
    MAX_ENTITIES = 32
    feats = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
    etype = np.zeros(MAX_ENTITIES, dtype=np.int64)
    mask  = np.zeros(MAX_ENTITIES, dtype=np.int8)

    slot = 0

    def _add_entity(type_id: int, dist_norm: float = 1.0, sin_a: float = 0.0, cos_a: float = 1.0):
        nonlocal slot
        if slot >= MAX_ENTITIES:
            return
        feats[slot, 0] = dist_norm
        feats[slot, 1] = sin_a
        feats[slot, 2] = cos_a
        etype[slot] = type_id
        mask[slot] = 1
        slot += 1

    for _ in range(n_enemies):
        _add_entity(1, dist_norm=1.0)   # enemy_melee, dist 1.0 (norm)
    for _ in range(n_keys):
        _add_entity(9, dist_norm=0.5)   # key
    for _ in range(n_health):
        _add_entity(5, dist_norm=0.8)   # health pack
    for _ in range(n_projectiles):
        _add_entity(4, dist_norm=0.3)   # incoming projectile

    # --- vitals ---
    vitals = np.zeros(17, dtype=np.float32)
    vitals[0] = health
    vitals[8] = 1.0 if has_blue_card else 0.0  # blue_card flag

    return {
        "rgb":   np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.zeros((120, 160),    dtype=np.uint8),
        "vitals": vitals,
        "rays":  {"distance": distances, "hit_type": hit_types},
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


ANALYZER = SpatialAnalyzer()   # stateless — safe to share across tests


# ===========================================================================
# SECTION 1: ray_summary
# ===========================================================================

class TestRaySummary:

    def test_all_open_obs(self):
        obs = _make_obs()  # all hit_types=0 (none)
        s = ANALYZER.ray_summary(obs)
        assert isinstance(s, RaySummary)
        assert s.n_open == NUM_RAYS
        assert s.n_wall == 0
        assert s.n_door == 0
        assert s.n_locked == 0
        assert s.n_enemy == 0
        assert s.n_item == 0

    def test_counts_one_wall_ray(self):
        obs = _make_obs(ray_hit_types=[_RAY_HIT_WALL])
        s = ANALYZER.ray_summary(obs)
        assert s.n_wall == 1
        assert s.n_open == NUM_RAYS - 1

    def test_counts_door_and_locked(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][5] = _RAY_HIT_DOOR
        obs["rays"]["hit_type"][10] = _RAY_HIT_LOCKED
        obs["rays"]["hit_type"][20] = _RAY_HIT_LOCKED
        s = ANALYZER.ray_summary(obs)
        assert s.n_door == 1
        assert s.n_locked == 2

    def test_counts_enemy_and_item_rays(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][0] = _RAY_HIT_ENEMY
        obs["rays"]["hit_type"][1] = _RAY_HIT_ITEM
        obs["rays"]["hit_type"][2] = _RAY_HIT_ITEM
        s = ANALYZER.ray_summary(obs)
        assert s.n_enemy == 1
        assert s.n_item == 2

    def test_min_dist_is_smallest_ray(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 1.0
        obs["rays"]["distance"][15] = 0.1   # closest
        s = ANALYZER.ray_summary(obs)
        assert s.min_dist_norm == pytest.approx(0.1)

    def test_min_dist_angle_corresponds_to_closest_ray(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 1.0
        obs["rays"]["distance"][8] = 0.05  # ray 8 = 45°
        s = ANALYZER.ray_summary(obs)
        expected_angle = 8 * (360.0 / NUM_RAYS)
        assert s.min_dist_angle == pytest.approx(expected_angle)

    def test_mean_dist_matches_numpy(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 0.5
        obs["rays"]["distance"][0] = 0.1
        s = ANALYZER.ray_summary(obs)
        expected = float(obs["rays"]["distance"].mean())
        assert s.mean_dist_norm == pytest.approx(expected, rel=1e-5)

    def test_total_counts_sum_to_num_rays(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][0:4]   = _RAY_HIT_WALL
        obs["rays"]["hit_type"][4:6]   = _RAY_HIT_DOOR
        obs["rays"]["hit_type"][6]     = _RAY_HIT_ENEMY
        s = ANALYZER.ray_summary(obs)
        total = s.n_wall + s.n_door + s.n_locked + s.n_enemy + s.n_item + s.n_open
        assert total == NUM_RAYS

    def test_all_walls(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][:] = _RAY_HIT_WALL
        s = ANALYZER.ray_summary(obs)
        assert s.n_wall == NUM_RAYS
        assert s.n_open == 0


# ===========================================================================
# SECTION 2: nearest_threat
# ===========================================================================

class TestNearestThreat:

    def test_no_enemies_returns_absent(self):
        obs = _make_obs(n_enemies=0)
        t = ANALYZER.nearest_threat(obs)
        assert isinstance(t, ThreatReport)
        assert t.present is False

    def test_single_enemy_detected(self):
        obs = _make_obs(n_enemies=1)
        t = ANALYZER.nearest_threat(obs)
        assert t.present is True

    def test_nearest_of_two_enemies(self):
        obs = _make_obs()
        ve = obs["visible_entities"]
        # Slot 0: closer enemy
        ve["features"][0, 0] = 0.3
        ve["entity_type"][0] = 1   # melee
        ve["mask"][0] = 1
        # Slot 1: farther enemy
        ve["features"][1, 0] = 1.5
        ve["entity_type"][1] = 2   # hitscan
        ve["mask"][1] = 1
        t = ANALYZER.nearest_threat(obs)
        assert t.present is True
        assert t.dist_norm == pytest.approx(0.3)
        assert t.type_id == 1

    def test_threat_dist_units_matches_norm(self):
        obs = _make_obs(n_enemies=1)
        obs["visible_entities"]["features"][0, 0] = 2.0
        t = ANALYZER.nearest_threat(obs)
        assert t.dist_units == pytest.approx(2.0 * RAY_RANGE)

    def test_threat_type_flags_melee(self):
        obs = _make_obs()
        obs["visible_entities"]["features"][0, 0] = 1.0
        obs["visible_entities"]["entity_type"][0] = 1   # melee
        obs["visible_entities"]["mask"][0] = 1
        t = ANALYZER.nearest_threat(obs)
        assert t.is_melee is True
        assert t.is_hitscan is False
        assert t.is_projectile_enemy is False

    def test_threat_type_flags_hitscan(self):
        obs = _make_obs()
        obs["visible_entities"]["entity_type"][0] = 2   # hitscan
        obs["visible_entities"]["mask"][0] = 1
        t = ANALYZER.nearest_threat(obs)
        assert t.is_hitscan is True
        assert t.is_melee is False

    def test_threat_type_flags_projectile_enemy(self):
        obs = _make_obs()
        obs["visible_entities"]["entity_type"][0] = 3   # projectile enemy
        obs["visible_entities"]["mask"][0] = 1
        t = ANALYZER.nearest_threat(obs)
        assert t.is_projectile_enemy is True

    def test_sin_cos_angle_propagated(self):
        obs = _make_obs()
        sin_a = math.sin(math.radians(45.0))
        cos_a = math.cos(math.radians(45.0))
        obs["visible_entities"]["features"][0, 1] = sin_a
        obs["visible_entities"]["features"][0, 2] = cos_a
        obs["visible_entities"]["entity_type"][0] = 1
        obs["visible_entities"]["mask"][0] = 1
        t = ANALYZER.nearest_threat(obs)
        assert t.sin_angle == pytest.approx(sin_a, rel=1e-5)
        assert t.cos_angle == pytest.approx(cos_a, rel=1e-5)

    def test_non_enemy_entities_ignored(self):
        """Key (type=9) and health (type=5) must not trigger a threat report."""
        obs = _make_obs(n_keys=1, n_health=1)
        t = ANALYZER.nearest_threat(obs)
        assert t.present is False

    def test_masked_out_enemies_not_counted(self):
        obs = _make_obs()
        obs["visible_entities"]["entity_type"][0] = 1
        obs["visible_entities"]["mask"][0] = 0  # explicitly unmasked
        t = ANALYZER.nearest_threat(obs)
        assert t.present is False


# ===========================================================================
# SECTION 3: nearest_pickup
# ===========================================================================

class TestNearestPickup:

    def test_no_pickups_returns_absent(self):
        obs = _make_obs(n_enemies=0)
        p = ANALYZER.nearest_pickup(obs)
        assert isinstance(p, PickupReport)
        assert p.present is False

    def test_key_takes_priority_over_health(self):
        """Key (type_id=9) must beat health (type_id=5) in default priority."""
        obs = _make_obs(n_keys=1, n_health=1)
        # Both visible; key dist = 0.5, health dist = 0.8 (from fixture)
        p = ANALYZER.nearest_pickup(obs)
        assert p.present is True
        assert p.type_id == 9
        assert p.type_name == "key"

    def test_health_returned_when_no_key(self):
        obs = _make_obs(n_health=1)
        p = ANALYZER.nearest_pickup(obs)
        assert p.present is True
        assert p.type_id == 5
        assert p.type_name == "health"

    def test_custom_priority_order_respected(self):
        """With priority=(5, 9), health should beat key."""
        obs = _make_obs(n_keys=1, n_health=1)
        p = ANALYZER.nearest_pickup(obs, priority=(5, 9))
        assert p.type_id == 5

    def test_pickup_dist_units_matches_norm(self):
        obs = _make_obs(n_health=1)
        obs["visible_entities"]["features"][0, 0] = 1.5
        p = ANALYZER.nearest_pickup(obs)
        assert p.dist_units == pytest.approx(1.5 * RAY_RANGE)

    def test_enemies_not_returned_as_pickups(self):
        obs = _make_obs(n_enemies=3)
        p = ANALYZER.nearest_pickup(obs)
        assert p.present is False


# ===========================================================================
# SECTION 4: door_report
# ===========================================================================

class TestDoorReport:

    def test_no_doors_all_false(self):
        obs = _make_obs()  # all rays open
        d = ANALYZER.door_report(obs)
        assert isinstance(d, DoorReport)
        assert d.has_door is False
        assert d.has_locked is False
        assert d.n_doors == 0
        assert d.n_locked == 0

    def test_detects_single_door(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][10] = _RAY_HIT_DOOR
        obs["rays"]["distance"][10] = 0.4
        d = ANALYZER.door_report(obs)
        assert d.has_door is True
        assert d.n_doors == 1
        assert d.nearest_door_dist_norm == pytest.approx(0.4)

    def test_detects_locked_door(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][20] = _RAY_HIT_LOCKED
        obs["rays"]["distance"][20] = 0.6
        d = ANALYZER.door_report(obs)
        assert d.has_locked is True
        assert d.n_locked == 1
        assert d.nearest_locked_dist_norm == pytest.approx(0.6)

    def test_nearest_door_selected_when_multiple(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][5]  = _RAY_HIT_DOOR
        obs["rays"]["distance"][5]  = 0.9
        obs["rays"]["hit_type"][10] = _RAY_HIT_DOOR
        obs["rays"]["distance"][10] = 0.2   # closer
        d = ANALYZER.door_report(obs)
        assert d.n_doors == 2
        assert d.nearest_door_dist_norm == pytest.approx(0.2)

    def test_door_and_locked_simultaneously(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][3]  = _RAY_HIT_DOOR
        obs["rays"]["hit_type"][15] = _RAY_HIT_LOCKED
        d = ANALYZER.door_report(obs)
        assert d.has_door is True
        assert d.has_locked is True

    def test_door_angle_matches_ray_index(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][16] = _RAY_HIT_DOOR   # ray 16 = 90°
        d = ANALYZER.door_report(obs)
        expected_angle = 16 * (360.0 / NUM_RAYS)
        assert d.nearest_door_angle_deg == pytest.approx(expected_angle)


# ===========================================================================
# SECTION 5: navigation_hints
# ===========================================================================

class TestNavigationHints:

    def test_all_open_obs_full_clearance(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 1.0   # all rays fully open
        n = ANALYZER.navigation_hints(obs)
        assert isinstance(n, NavigationHints)
        assert n.forward_clearance == pytest.approx(1.0)
        assert n.left_clearance == pytest.approx(1.0)
        assert n.right_clearance == pytest.approx(1.0)

    def test_blocked_forward_reduces_forward_clearance(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 1.0
        # Block all forward rays (0,1,2,3,61,62,63) below threshold
        for i in [0, 1, 2, 3, 61, 62, 63]:
            obs["rays"]["distance"][i] = 0.1
        n = ANALYZER.navigation_hints(obs)
        assert n.forward_clearance == pytest.approx(0.0)

    def test_best_escape_ray_is_max_distance_ray(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 0.1
        obs["rays"]["distance"][32] = 1.0  # ray directly behind = best escape
        n = ANALYZER.navigation_hints(obs)
        assert n.best_escape_ray == 32
        assert n.best_escape_dist == pytest.approx(1.0)

    def test_rear_clearance_computed(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 1.0
        # Block all rear rays (28..36) completely
        obs["rays"]["distance"][28:36] = 0.1
        n = ANALYZER.navigation_hints(obs, open_threshold=0.9)
        assert n.rear_clearance == pytest.approx(0.0)

    def test_custom_threshold(self):
        obs = _make_obs()
        obs["rays"]["distance"][:] = 0.5
        n_strict = ANALYZER.navigation_hints(obs, open_threshold=0.9)
        n_loose  = ANALYZER.navigation_hints(obs, open_threshold=0.3)
        # With loose threshold, distances of 0.5 count as open
        assert n_loose.forward_clearance >= n_strict.forward_clearance


# ===========================================================================
# SECTION 6: key_status
# ===========================================================================

class TestKeyStatus:

    def test_no_keys_all_false(self):
        obs = _make_obs()
        ks = ANALYZER.key_status(obs)
        assert isinstance(ks, dict)
        assert all(not v for v in ks.values())

    def test_blue_card_detected(self):
        obs = _make_obs(has_blue_card=True)
        ks = ANALYZER.key_status(obs)
        assert ks["blue_card"] is True

    def test_multiple_keys(self):
        obs = _make_obs()
        obs["vitals"][8]  = 1.0   # blue_card
        obs["vitals"][10] = 1.0   # red_card
        obs["vitals"][13] = 1.0   # red_skull
        ks = ANALYZER.key_status(obs)
        assert ks["blue_card"] is True
        assert ks["red_card"] is True
        assert ks["red_skull"] is True
        assert ks["yellow_card"] is False
        assert ks["blue_skull"] is False
        assert ks["yellow_skull"] is False

    def test_key_names_present(self):
        obs = _make_obs()
        ks = ANALYZER.key_status(obs)
        expected = {"blue_card", "yellow_card", "red_card",
                    "blue_skull", "yellow_skull", "red_skull"}
        assert set(ks.keys()) == expected


# ===========================================================================
# SECTION 7: has_incoming_projectile
# ===========================================================================

class TestHasIncomingProjectile:

    def test_no_projectile(self):
        obs = _make_obs(n_enemies=1)
        assert ANALYZER.has_incoming_projectile(obs) is False

    def test_projectile_detected(self):
        obs = _make_obs(n_projectiles=1)
        assert ANALYZER.has_incoming_projectile(obs) is True

    def test_projectile_in_later_slot(self):
        obs = _make_obs(n_enemies=2, n_projectiles=1)
        assert ANALYZER.has_incoming_projectile(obs) is True

    def test_masked_out_projectile_not_counted(self):
        obs = _make_obs()
        obs["visible_entities"]["entity_type"][0] = 4   # projectile
        obs["visible_entities"]["mask"][0] = 0           # masked
        assert ANALYZER.has_incoming_projectile(obs) is False


# ===========================================================================
# SECTION 8: full_report
# ===========================================================================

class TestFullReport:

    def test_full_report_has_all_expected_keys(self):
        obs = _make_obs(n_enemies=1, n_keys=1)
        report = ANALYZER.full_report(obs)
        expected_keys = {
            "ray_summary", "nearest_threat", "nearest_pickup",
            "door_report", "nav_hints", "has_projectile", "keys_held",
        }
        assert set(report.keys()) == expected_keys

    def test_full_report_does_not_crash_on_empty_obs(self):
        obs = _make_obs()
        report = ANALYZER.full_report(obs)
        assert report is not None

    def test_full_report_all_values_are_serialisable(self):
        """All values in the report must be plain Python types (bool, int, float, dict)."""
        obs = _make_obs(n_enemies=1, n_keys=1, n_health=1)
        report = ANALYZER.full_report(obs)
        # Flatten one level and check types
        import json
        try:
            json.dumps(report)
        except TypeError as e:
            pytest.fail(f"full_report contains non-serialisable values: {e}")

    def test_full_report_ray_summary_counts_match(self):
        obs = _make_obs()
        obs["rays"]["hit_type"][0] = _RAY_HIT_ENEMY
        obs["rays"]["hit_type"][1] = _RAY_HIT_WALL
        report = ANALYZER.full_report(obs)
        assert report["ray_summary"]["n_enemy"] == 1
        assert report["ray_summary"]["n_wall"] == 1

    def test_full_report_does_not_modify_obs(self):
        obs = _make_obs(n_enemies=1)
        orig_dist = obs["rays"]["distance"].copy()
        ANALYZER.full_report(obs)
        np.testing.assert_array_equal(obs["rays"]["distance"], orig_dist)

    def test_analyzer_is_reusable(self):
        """SpatialAnalyzer should produce identical results when called twice."""
        obs = _make_obs(n_enemies=2, n_keys=1)
        r1 = ANALYZER.full_report(obs)
        r2 = ANALYZER.full_report(obs)
        # Threat dist should match
        assert r1["nearest_threat"]["dist_norm"] == pytest.approx(r2["nearest_threat"]["dist_norm"])


# ---------------------------------------------------------------------------
# Allow running directly: python perception/tests/test_spatial_analyzer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    test_classes = [
        TestRaySummary,
        TestNearestThreat,
        TestNearestPickup,
        TestDoorReport,
        TestNavigationHints,
        TestKeyStatus,
        TestHasIncomingProjectile,
        TestFullReport,
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
