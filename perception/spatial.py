"""Spatial intelligence and ego-centric perception processing.

Extracts geometric corridors, door detection, entity kinematics,
and normalized sensor features from ViZDoom observation dictionaries.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


class SpatialPerception:
    """Ego-centric spatial reasoning and observation processing."""

    RAY_RANGE: float = 1024.0
    NUM_RAYS: int = 64
    MAX_ENTITIES: int = 32

    # Hit types from CONTRACT.md: 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item
    HIT_NONE = 0
    HIT_WALL = 1
    HIT_DOOR = 2
    HIT_LOCKED_DOOR = 3
    HIT_ENEMY = 4
    HIT_ITEM = 5

    def __init__(self) -> None:
        # Precomputed ray angles (radians): ray 0 straight ahead, CCW
        self.ray_angles = np.linspace(0, 2 * np.pi, self.NUM_RAYS, endpoint=False, dtype=np.float32)

    def process_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and prepare observation fields for model consumption.
        
        Args:
            obs: Observation dict conforming to CONTRACT.md
        Returns:
            Dictionary with verified vitals, rays, and visible_entities structures.
        """
        vitals = np.asarray(obs["vitals"], dtype=np.float32)
        rays_dist = np.asarray(obs["rays"]["distance"], dtype=np.float32)
        rays_hit = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)

        ent = obs["visible_entities"]
        ent_feats = np.asarray(ent["features"], dtype=np.float32)
        ent_type = np.asarray(ent["entity_type"], dtype=np.int64)
        ent_mask = np.asarray(ent["mask"], dtype=np.float32)

        return {
            "vitals": vitals,
            "rays": {
                "distance": rays_dist,
                "hit_type": rays_hit,
            },
            "visible_entities": {
                "features": ent_feats,
                "entity_type": ent_type,
                "mask": ent_mask,
            },
        }

    def detect_doors(self, ray_dist: np.ndarray, ray_hit: np.ndarray) -> dict[str, Any]:
        """Identify presence, angle, and distance to normal and locked doors.
        
        Returns:
            dict containing:
              - 'has_door': bool
              - 'has_locked_door': bool
              - 'door_dist': float (min distance, or 1.0)
              - 'door_ray_idx': int or None
              - 'locked_door_dist': float
              - 'locked_door_ray_idx': int or None
        """
        door_indices = np.where(ray_hit == self.HIT_DOOR)[0]
        locked_indices = np.where(ray_hit == self.HIT_LOCKED_DOOR)[0]

        res: dict[str, Any] = {
            "has_door": len(door_indices) > 0,
            "has_locked_door": len(locked_indices) > 0,
            "door_dist": float(np.min(ray_dist[door_indices])) if len(door_indices) > 0 else 1.0,
            "door_ray_idx": int(door_indices[np.argmin(ray_dist[door_indices])]) if len(door_indices) > 0 else None,
            "locked_door_dist": float(np.min(ray_dist[locked_indices])) if len(locked_indices) > 0 else 1.0,
            "locked_door_ray_idx": int(locked_indices[np.argmin(ray_dist[locked_indices])]) if len(locked_indices) > 0 else None,
        }
        return res

    def get_forward_clearance(self, ray_dist: np.ndarray, window_rays: int = 5) -> float:
        """Measure clearance directly ahead of player across a narrow cone."""
        # Ray 0 is forward, rays 1..window_rays are left, rays 63..64-window_rays are right
        left_slice = ray_dist[:window_rays]
        right_slice = ray_dist[-window_rays:]
        forward_rays = np.concatenate([left_slice, right_slice])
        return float(np.mean(forward_rays))

    def get_nearest_threat(self, ent_feats: np.ndarray, ent_mask: np.ndarray) -> dict[str, float] | None:
        """Return distance and angle to nearest active entity if any exist."""
        active_indices = np.where(ent_mask > 0.5)[0]
        if len(active_indices) == 0:
            return None

        # Feature layout: [dist / 1024, sin(rel_ang), cos(rel_ang), rel_vel_fwd, rel_vel_right]
        distances = ent_feats[active_indices, 0]
        nearest_idx = active_indices[np.argmin(distances)]
        feat = ent_feats[nearest_idx]

        return {
            "norm_distance": float(feat[0]),
            "sin_angle": float(feat[1]),
            "cos_angle": float(feat[2]),
            "rel_vel_fwd": float(feat[3]),
            "rel_vel_right": float(feat[4]),
        }
