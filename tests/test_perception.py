"""Unit tests for perception module and spatial reasoning."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from perception.spatial import SpatialPerception


def test_spatial_perception_process_observation():
    perc = SpatialPerception()
    sample_obs = {
        "vitals": np.zeros(17, dtype=np.float32),
        "rays": {
            "distance": np.ones(64, dtype=np.float32),
            "hit_type": np.zeros(64, dtype=np.int64),
        },
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros(32, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.float32),
        },
    }
    processed = perc.process_observation(sample_obs)
    assert processed["vitals"].shape == (17,)
    assert processed["rays"]["distance"].shape == (64,)
    assert processed["rays"]["hit_type"].shape == (64,)
    assert processed["visible_entities"]["features"].shape == (32, 5)
    assert processed["visible_entities"]["entity_type"].shape == (32,)
    assert processed["visible_entities"]["mask"].shape == (32,)


def test_door_detection():
    perc = SpatialPerception()
    ray_dist = np.ones(64, dtype=np.float32)
    ray_hit = np.ones(64, dtype=np.int64)  # all walls

    # No doors
    res = perc.detect_doors(ray_dist, ray_hit)
    assert not res["has_door"]
    assert not res["has_locked_door"]
    assert res["door_dist"] == 1.0

    # Normal door at ray 0
    ray_hit[0] = SpatialPerception.HIT_DOOR
    ray_dist[0] = 0.25
    res = perc.detect_doors(ray_dist, ray_hit)
    assert res["has_door"]
    assert res["door_dist"] == pytest.approx(0.25)
    assert res["door_ray_idx"] == 0

    # Locked door at ray 16
    ray_hit[16] = SpatialPerception.HIT_LOCKED_DOOR
    ray_dist[16] = 0.45
    res = perc.detect_doors(ray_dist, ray_hit)
    assert res["has_locked_door"]
    assert res["locked_door_dist"] == pytest.approx(0.45)
    assert res["locked_door_ray_idx"] == 16


def test_nearest_threat_empty():
    perc = SpatialPerception()
    ent_feats = np.zeros((32, 5), dtype=np.float32)
    ent_mask = np.zeros(32, dtype=np.float32)

    threat = perc.get_nearest_threat(ent_feats, ent_mask)
    assert threat is None


def test_nearest_threat_active():
    perc = SpatialPerception()
    ent_feats = np.zeros((32, 5), dtype=np.float32)
    ent_mask = np.zeros(32, dtype=np.float32)

    # Threat 1 at distance 0.8
    ent_feats[0, 0] = 0.8
    ent_feats[0, 1] = 0.5  # sin
    ent_feats[0, 2] = 0.866  # cos
    ent_mask[0] = 1.0

    # Threat 2 at distance 0.3 (closer)
    ent_feats[1, 0] = 0.3
    ent_feats[1, 1] = -0.2
    ent_feats[1, 2] = 0.98
    ent_mask[1] = 1.0

    threat = perc.get_nearest_threat(ent_feats, ent_mask)
    assert threat is not None
    assert threat["norm_distance"] == pytest.approx(0.3)
    assert threat["sin_angle"] == pytest.approx(-0.2)
