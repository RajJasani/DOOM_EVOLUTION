"""SpatialAnalyzer — ego-centric spatial statistics from raw observations.

Provides fast NumPy utilities for analysing the agent's current spatial
situation from a single observation step. These helpers are useful for:
  - Reward shaping during training (Member 4)
  - Feature extraction for the visual encoder (Member 2)
  - Debugging spatial understanding (everyone)

Design rules
------------
- Pure NumPy, no torch dependency.
- All inputs are raw obs fields — do NOT normalise before calling these.
- Results are returned as plain Python dicts with float/int values and
  numpy arrays; easy to log, assert against, or convert to tensors.
- Every function is stateless (no side effects, no instance state).
  Spatial *memory* (accumulation across steps) lives in SpatialMemory,
  which is a separate class (to be implemented in Phase 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAY_HIT_NONE   = 0
_RAY_HIT_WALL   = 1
_RAY_HIT_DOOR   = 2
_RAY_HIT_LOCKED = 3
_RAY_HIT_ENEMY  = 4
_RAY_HIT_ITEM   = 5

_ENTITY_TYPE_NAMES = (
    "pad", "enemy_melee", "enemy_hitscan", "enemy_projectile",
    "projectile", "health", "armor", "ammo", "weapon", "key", "barrel", "other",
)
_ENEMY_TYPE_IDS   = frozenset({1, 2, 3})       # melee, hitscan, projectile
_PICKUP_TYPE_IDS  = frozenset({5, 6, 7, 8})    # health, armor, ammo, weapon
_KEY_TYPE_ID      = 9
_PROJECTILE_ID    = 4

NUM_RAYS      = 64
RAY_RANGE     = 1024.0
ENTITY_DIST_MAX_NORM = 4.0  # = 4096 / 1024

# How many consecutive silent (open) rays constitute a navigation gap?
_OPEN_CORRIDOR_MIN_WIDTH = 3  # ~17° arc


# ---------------------------------------------------------------------------
# Named result types
# ---------------------------------------------------------------------------

class RaySummary(NamedTuple):
    """Aggregate statistics over all 64 rays."""
    n_wall:   int    # rays that hit a wall
    n_door:   int    # rays that hit a closed door
    n_locked: int    # rays that hit a locked door
    n_enemy:  int    # rays that hit an enemy
    n_item:   int    # rays that hit an item
    n_open:   int    # rays that hit nothing within range (clear path)
    mean_dist_norm: float   # mean normalised distance across all rays [0,1]
    min_dist_norm:  float   # closest hit distance [0,1]
    min_dist_angle: float   # angle (degrees CCW from forward) of closest hit


@dataclass
class ThreatReport:
    """Summary of the most dangerous entity visible."""
    present:    bool  = False
    type_name:  str   = "none"
    type_id:    int   = 0
    dist_norm:  float = 0.0   # normalised (/ 1024)
    dist_units: float = 0.0   # map units
    angle_deg:  float = 0.0   # +left, -right, 0=ahead
    sin_angle:  float = 0.0
    cos_angle:  float = 0.0
    rel_vel_forward: float = 0.0
    rel_vel_right:   float = 0.0
    is_melee:   bool  = False
    is_hitscan: bool  = False
    is_projectile_enemy: bool = False


@dataclass
class PickupReport:
    """Summary of the highest-priority visible pickup."""
    present:   bool  = False
    type_name: str   = "none"
    type_id:   int   = 0
    dist_norm: float = 0.0
    dist_units: float = 0.0
    angle_deg: float = 0.0


@dataclass
class DoorReport:
    """Doors detected by rays (may not be in FOV)."""
    has_door:   bool  = False
    has_locked: bool  = False
    n_doors:    int   = 0
    n_locked:   int   = 0
    nearest_door_dist_norm:   float = 1.0  # 1.0 = nothing
    nearest_door_angle_deg:   float = 0.0
    nearest_locked_dist_norm: float = 1.0
    nearest_locked_angle_deg: float = 0.0


@dataclass
class NavigationHints:
    """Suggested movement directions derived from ray openings."""
    # Fraction of forward arc (±22.5°) that is open
    forward_clearance: float = 0.0
    # Fraction of left arc (45°–135°) that is open
    left_clearance: float    = 0.0
    # Fraction of right arc (225°–315°) that is open
    right_clearance: float   = 0.0
    # Fraction of rear arc (135°–225°) that is open
    rear_clearance: float    = 0.0
    # Indices of rays with the most open space (best escape directions)
    best_escape_ray: int     = 0
    best_escape_dist: float  = 1.0


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class SpatialAnalyzer:
    """Stateless spatial analysis of a single DOOM observation.

    Instantiate once, reuse across steps:
        analyzer = SpatialAnalyzer()
        report = analyzer.full_report(obs)

    All results are computed purely from the current obs dict (no history).
    For cross-step spatial memory, see SpatialMemory (Phase 4).
    """

    # ------------------------------------------------------------------
    def ray_summary(self, obs: dict) -> RaySummary:
        """Aggregate statistics over the 64 rays."""
        rays = obs["rays"]
        distance = np.asarray(rays["distance"], dtype=np.float32)
        hit_type = np.asarray(rays["hit_type"], dtype=np.int64)

        angles_deg = np.arange(NUM_RAYS) * (360.0 / NUM_RAYS)

        n_wall   = int((hit_type == _RAY_HIT_WALL).sum())
        n_door   = int((hit_type == _RAY_HIT_DOOR).sum())
        n_locked = int((hit_type == _RAY_HIT_LOCKED).sum())
        n_enemy  = int((hit_type == _RAY_HIT_ENEMY).sum())
        n_item   = int((hit_type == _RAY_HIT_ITEM).sum())
        n_open   = int((hit_type == _RAY_HIT_NONE).sum())

        mean_d = float(distance.mean())
        min_idx = int(np.argmin(distance))
        min_d   = float(distance[min_idx])
        min_ang = float(angles_deg[min_idx])

        return RaySummary(
            n_wall=n_wall, n_door=n_door, n_locked=n_locked,
            n_enemy=n_enemy, n_item=n_item, n_open=n_open,
            mean_dist_norm=mean_d, min_dist_norm=min_d,
            min_dist_angle=min_ang,
        )

    # ------------------------------------------------------------------
    def nearest_threat(self, obs: dict) -> ThreatReport:
        """Return a report about the nearest visible enemy."""
        ve    = obs["visible_entities"]
        feats = np.asarray(ve["features"],     dtype=np.float32)
        etype = np.asarray(ve["entity_type"],  dtype=np.int64)
        mask  = np.asarray(ve["mask"],         dtype=np.int8)

        # Filter to real enemy slots, sorted nearest first (they're already sorted)
        is_enemy = np.isin(etype, list(_ENEMY_TYPE_IDS))
        active   = (mask == 1) & is_enemy

        if not active.any():
            return ThreatReport(present=False)

        idx = int(np.flatnonzero(active)[0])  # nearest (entities sorted by dist)
        d_norm = float(feats[idx, 0])
        sin_a  = float(feats[idx, 1])
        cos_a  = float(feats[idx, 2])
        vf     = float(feats[idx, 3])
        vr     = float(feats[idx, 4])
        angle_deg = float(math.degrees(math.atan2(sin_a, cos_a)))
        tid   = int(etype[idx])

        return ThreatReport(
            present=True,
            type_name=_ENTITY_TYPE_NAMES[tid],
            type_id=tid,
            dist_norm=d_norm,
            dist_units=d_norm * RAY_RANGE,
            angle_deg=angle_deg,
            sin_angle=sin_a,
            cos_angle=cos_a,
            rel_vel_forward=vf,
            rel_vel_right=vr,
            is_melee=(tid == 1),
            is_hitscan=(tid == 2),
            is_projectile_enemy=(tid == 3),
        )

    # ------------------------------------------------------------------
    def nearest_pickup(self, obs: dict, priority: tuple[int, ...] = (9, 5, 6, 7, 8)) -> PickupReport:
        """Return the highest-priority visible pickup.

        Parameters
        ----------
        priority : tuple[int, ...]
            Entity type IDs in order of importance. Default: key > health >
            armor > ammo > weapon.
        """
        ve    = obs["visible_entities"]
        feats = np.asarray(ve["features"],    dtype=np.float32)
        etype = np.asarray(ve["entity_type"], dtype=np.int64)
        mask  = np.asarray(ve["mask"],        dtype=np.int8)

        for tid in priority:
            matches = (mask == 1) & (etype == tid)
            if matches.any():
                idx = int(np.flatnonzero(matches)[0])
                d_norm = float(feats[idx, 0])
                sin_a  = float(feats[idx, 1])
                cos_a  = float(feats[idx, 2])
                angle_deg = float(math.degrees(math.atan2(sin_a, cos_a)))
                return PickupReport(
                    present=True,
                    type_name=_ENTITY_TYPE_NAMES[tid],
                    type_id=tid,
                    dist_norm=d_norm,
                    dist_units=d_norm * RAY_RANGE,
                    angle_deg=angle_deg,
                )
        return PickupReport(present=False)

    # ------------------------------------------------------------------
    def door_report(self, obs: dict) -> DoorReport:
        """Identify doors and locked doors from ray data (full 360°)."""
        rays     = obs["rays"]
        distance = np.asarray(rays["distance"], dtype=np.float32)
        hit_type = np.asarray(rays["hit_type"], dtype=np.int64)
        angles_deg = np.arange(NUM_RAYS) * (360.0 / NUM_RAYS)

        door_mask   = hit_type == _RAY_HIT_DOOR
        locked_mask = hit_type == _RAY_HIT_LOCKED

        report = DoorReport(
            has_door=bool(door_mask.any()),
            has_locked=bool(locked_mask.any()),
            n_doors=int(door_mask.sum()),
            n_locked=int(locked_mask.sum()),
        )

        if report.has_door:
            door_dists = np.where(door_mask, distance, np.inf)
            best_door = int(np.argmin(door_dists))
            report.nearest_door_dist_norm = float(distance[best_door])
            report.nearest_door_angle_deg = float(angles_deg[best_door])

        if report.has_locked:
            locked_dists = np.where(locked_mask, distance, np.inf)
            best_locked = int(np.argmin(locked_dists))
            report.nearest_locked_dist_norm = float(distance[best_locked])
            report.nearest_locked_angle_deg = float(angles_deg[best_locked])

        return report

    # ------------------------------------------------------------------
    def navigation_hints(self, obs: dict, open_threshold: float = 0.9) -> NavigationHints:
        """Derive movement clearance from ray patterns.

        Parameters
        ----------
        open_threshold : float
            A ray is considered "open" (passable) if its normalised distance
            exceeds this value. Default 0.9 = at least 921 map units clear.
        """
        rays     = obs["rays"]
        distance = np.asarray(rays["distance"], dtype=np.float32)
        hit_type = np.asarray(rays.get("hit_type", np.zeros_like(distance)), dtype=np.int64)

        # A ray is open if distance >= open_threshold AND (hit_type == _RAY_HIT_NONE or distance is large)
        # However, if distance is blocked (< open_threshold), it is definitely NOT open regardless of hit_type.
        is_open  = (distance >= open_threshold)

        def _arc_fraction(start_ray: int, end_ray: int) -> float:
            """Fraction of rays in arc [start_ray, end_ray) that are open."""
            indices = np.arange(start_ray, end_ray) % NUM_RAYS
            return float(is_open[indices].mean())

        # Forward: rays 0,1,2,3,61,62,63 (±~20°) — 7 rays
        forward_rays = list(range(0, 4)) + list(range(61, 64))
        forward_clearance = float(is_open[forward_rays].mean())

        # Left quadrant: rays 8-23 (45°-135°)
        left_clearance  = _arc_fraction(8, 24)
        # Right quadrant: rays 40-55 (225°-315°)
        right_clearance = _arc_fraction(40, 56)
        # Rear: rays 28-36 (157°-202°)
        rear_clearance  = _arc_fraction(28, 36)

        # Best single escape direction: ray with greatest open distance
        best_ray  = int(np.argmax(distance))
        best_dist = float(distance[best_ray])

        return NavigationHints(
            forward_clearance=forward_clearance,
            left_clearance=left_clearance,
            right_clearance=right_clearance,
            rear_clearance=rear_clearance,
            best_escape_ray=best_ray,
            best_escape_dist=best_dist,
        )

    # ------------------------------------------------------------------
    def has_incoming_projectile(self, obs: dict) -> bool:
        """Return True if any visible entity is classified as an incoming projectile."""
        ve    = obs["visible_entities"]
        etype = np.asarray(ve["entity_type"], dtype=np.int64)
        mask  = np.asarray(ve["mask"],        dtype=np.int8)
        return bool(((mask == 1) & (etype == _PROJECTILE_ID)).any())

    # ------------------------------------------------------------------
    def key_status(self, obs: dict) -> dict[str, bool]:
        """Return held key flags from vitals as a readable dict."""
        vitals = np.asarray(obs["vitals"], dtype=np.float32)
        return {
            "blue_card":    bool(vitals[8]  > 0.5),
            "yellow_card":  bool(vitals[9]  > 0.5),
            "red_card":     bool(vitals[10] > 0.5),
            "blue_skull":   bool(vitals[11] > 0.5),
            "yellow_skull": bool(vitals[12] > 0.5),
            "red_skull":    bool(vitals[13] > 0.5),
        }

    # ------------------------------------------------------------------
    def full_report(self, obs: dict) -> dict:
        """Return all spatial analysis results as a single dict."""
        return {
            "ray_summary":    self.ray_summary(obs)._asdict(),
            "nearest_threat": vars(self.nearest_threat(obs)),
            "nearest_pickup": vars(self.nearest_pickup(obs)),
            "door_report":    vars(self.door_report(obs)),
            "nav_hints":      vars(self.navigation_hints(obs)),
            "has_projectile": self.has_incoming_projectile(obs),
            "keys_held":      self.key_status(obs),
        }
