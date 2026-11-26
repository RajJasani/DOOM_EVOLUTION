"""Sensory preprocessing for DOOM observations.

Responsibilities
----------------
1. validate_observation   – assert shapes, dtypes and value ranges match CONTRACT.md.
2. normalize_vitals_numpy – reference NumPy implementation matching VitalsEncoder.normalize_vitals().
3. normalize_entity_features – normalise the (32, 5) entity feature matrix before it
                               enters EnemyAttentionModule (audit Finding 3).
4. obs_to_device_tensors  – convert a raw obs dict to PyTorch tensors on a target device.
5. process_observation    – single entry point called from submission/agent.py:
                            enriches the obs dict with any extra keys (e.g. visual_embedding)
                            without modifying the original dict.

Design rules
------------
- Never modify the caller's obs dict in place; always return a shallow-copy with
  only the new keys added.
- Keep every function side-effect free and deterministic.
- Avoid Python loops over entity slots; use NumPy vectorised ops throughout.
- All operations must complete in << 1 ms on a single CPU thread.

Observation key reference  (env/doom_env/spaces.py + docs/CONTRACT.md)
-----------------------------------------------------------------------
obs["rgb"]                           (120, 160, 3)  uint8
obs["depth"]                         (120, 160)     uint8
obs["vitals"]                        (17,)          float32   see VITALS_FIELDS
obs["rays"]["distance"]              (64,)          float32   [0.0, 1.0]
obs["rays"]["hit_type"]              (64,)          int64     {0..5}
obs["visible_entities"]["features"]  (32, 5)        float32   distance in [0, 4.0]!
obs["visible_entities"]["entity_type"] (32,)        int64     {0..11}
obs["visible_entities"]["mask"]      (32,)          int8      {0, 1}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — the perception module must work even without torch/gymnasium
# so that unit tests can run on a plain Python 3.13 install.
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# Add env/ to path so we can import doom_env.spaces for constants.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "env"))

try:
    from doom_env import spaces as _S
    _SPACES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPACES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (inlined from doom_env.spaces so we don't hard-require the env package
# during unit tests that only test preprocessing math).
# ---------------------------------------------------------------------------

SCREEN_HEIGHT  = 120
SCREEN_WIDTH   = 160
NUM_RAYS       = 64
NUM_HIT_TYPES  = 6     # {0=none, 1=wall, 2=door, 3=locked, 4=enemy, 5=item}
MAX_ENTITIES   = 32
NUM_ENTITY_TYPES = 12  # {0=pad, 1..11 = real types}
ENTITY_FEATURES  = 5   # (distance, sin_angle, cos_angle, rel_vel_fwd, rel_vel_right)
NUM_VITALS     = 17
RAY_RANGE      = 1024.0
ENTITY_MAX_DISTANCE_NORM = 4096.0 / RAY_RANGE  # = 4.0  (max entity features[:,0])

# Vitals normalisation constants  (mirror of VitalsEncoder.normalize_vitals)
_HEALTH_ARMOR_SCALE  = 1.0 / 100.0   # normal range → [0, 2.0]
_WEAPON_SCALE        = 1.0 / 7.0     # weapon slot [0,7] → [0, 1]
_AMMO_LOG_SCALE      = 1.0 / 4.0     # log1p(ammo) / 4; covers bullets=400 → ~1.5
_VELOCITY_SCALE      = 1.0 / 20.0    # running ~12.5 u/tic; / 20 gives headroom

# Entity feature normalisation constants  (audit Finding 3)
_ENT_DIST_SCALE      = 1.0 / ENTITY_MAX_DISTANCE_NORM   # [0, 4.0] → [0, 1.0]
_ENT_VEL_CLIP        = 20.0                              # clip extreme projectile velocities
_ENT_VEL_SCALE       = 1.0 / _ENT_VEL_CLIP              # → [-1, 1]


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------

def validate_observation(obs: dict) -> list[str]:
    """Check that obs matches the CONTRACT.md specification.

    Parameters
    ----------
    obs : dict
        Raw observation from DoomEnv or agent.act().

    Returns
    -------
    list[str]
        List of violation strings. Empty list means the observation is valid.

    Notes
    -----
    This function never raises — it returns all violations so the caller can
    decide whether to log, warn or assert. Call it like:
        issues = validate_observation(obs)
        assert not issues, issues
    """
    issues: list[str] = []

    # --- Top-level keys ------------------------------------------------
    required_keys = {"rgb", "depth", "vitals", "rays", "visible_entities"}
    missing = required_keys - set(obs)
    if missing:
        issues.append(f"Missing top-level keys: {missing}")
        return issues  # cannot check sub-keys

    # --- rgb -----------------------------------------------------------
    rgb = obs["rgb"]
    if not isinstance(rgb, np.ndarray):
        issues.append(f"rgb must be np.ndarray, got {type(rgb)}")
    else:
        if rgb.shape != (SCREEN_HEIGHT, SCREEN_WIDTH, 3):
            issues.append(f"rgb shape {rgb.shape} != ({SCREEN_HEIGHT}, {SCREEN_WIDTH}, 3)")
        if rgb.dtype != np.uint8:
            issues.append(f"rgb dtype {rgb.dtype} != uint8")

    # --- depth ---------------------------------------------------------
    depth = obs["depth"]
    if not isinstance(depth, np.ndarray):
        issues.append(f"depth must be np.ndarray, got {type(depth)}")
    else:
        if depth.shape != (SCREEN_HEIGHT, SCREEN_WIDTH):
            issues.append(f"depth shape {depth.shape} != ({SCREEN_HEIGHT}, {SCREEN_WIDTH})")
        if depth.dtype != np.uint8:
            issues.append(f"depth dtype {depth.dtype} != uint8")

    # --- vitals --------------------------------------------------------
    vitals = obs["vitals"]
    if not isinstance(vitals, np.ndarray):
        issues.append(f"vitals must be np.ndarray, got {type(vitals)}")
    else:
        if vitals.shape != (NUM_VITALS,):
            issues.append(f"vitals shape {vitals.shape} != ({NUM_VITALS},)")
        if vitals.dtype != np.float32:
            issues.append(f"vitals dtype {vitals.dtype} != float32")
        if np.any(np.isnan(vitals)):
            issues.append("vitals contains NaN")
        if np.any(np.isinf(vitals)):
            issues.append("vitals contains Inf")
        # Key flags must be binary
        key_flags = vitals[8:14]
        if not np.all((key_flags == 0.0) | (key_flags == 1.0)):
            issues.append(f"vitals[8:14] key flags must be 0 or 1, got {key_flags}")

    # --- rays ----------------------------------------------------------
    rays = obs["rays"]
    if not isinstance(rays, dict):
        issues.append(f"rays must be dict, got {type(rays)}")
    else:
        ray_keys = {"distance", "hit_type"}
        missing_ray = ray_keys - set(rays)
        if missing_ray:
            issues.append(f"rays missing sub-keys: {missing_ray}")
        else:
            d = rays["distance"]
            h = rays["hit_type"]
            if not isinstance(d, np.ndarray):
                issues.append(f"rays.distance must be np.ndarray, got {type(d)}")
            elif d.shape != (NUM_RAYS,):
                issues.append(f"rays.distance shape {d.shape} != ({NUM_RAYS},)")
            elif d.dtype != np.float32:
                issues.append(f"rays.distance dtype {d.dtype} != float32")
            elif np.any((d < 0.0) | (d > 1.0)):
                bad = d[(d < 0.0) | (d > 1.0)]
                issues.append(f"rays.distance out of [0,1]: {bad[:3]}")

            if not isinstance(h, np.ndarray):
                issues.append(f"rays.hit_type must be np.ndarray, got {type(h)}")
            elif h.shape != (NUM_RAYS,):
                issues.append(f"rays.hit_type shape {h.shape} != ({NUM_RAYS},)")
            elif not np.issubdtype(h.dtype, np.integer):
                issues.append(f"rays.hit_type dtype {h.dtype} not integer")
            elif np.any((h < 0) | (h >= NUM_HIT_TYPES)):
                issues.append(f"rays.hit_type values out of [0,{NUM_HIT_TYPES-1}]")

    # --- visible_entities ----------------------------------------------
    ve = obs["visible_entities"]
    if not isinstance(ve, dict):
        issues.append(f"visible_entities must be dict, got {type(ve)}")
    else:
        ve_keys = {"features", "entity_type", "mask"}
        missing_ve = ve_keys - set(ve)
        if missing_ve:
            issues.append(f"visible_entities missing sub-keys: {missing_ve}")
        else:
            feats = ve["features"]
            etype = ve["entity_type"]
            mask  = ve["mask"]

            if not isinstance(feats, np.ndarray):
                issues.append(f"visible_entities.features must be np.ndarray, got {type(feats)}")
            elif feats.shape != (MAX_ENTITIES, ENTITY_FEATURES):
                issues.append(f"features shape {feats.shape} != ({MAX_ENTITIES}, {ENTITY_FEATURES})")
            elif feats.dtype != np.float32:
                issues.append(f"features dtype {feats.dtype} != float32")
            elif np.any(np.isnan(feats)):
                issues.append("visible_entities.features contains NaN")
            else:
                # Distance column must be in [0, ENTITY_MAX_DISTANCE_NORM] for real slots
                m = (mask == 1) if isinstance(mask, np.ndarray) else np.ones(MAX_ENTITIES, bool)
                dist_col = feats[m.astype(bool), 0]
                if len(dist_col) > 0 and np.any(dist_col > ENTITY_MAX_DISTANCE_NORM + 1e-4):
                    bad_max = dist_col.max()
                    issues.append(
                        f"features[:,0] (distance) max={bad_max:.3f} > "
                        f"theoretical max {ENTITY_MAX_DISTANCE_NORM:.1f} (4096/1024)"
                    )

            if not isinstance(etype, np.ndarray):
                issues.append(f"entity_type must be np.ndarray, got {type(etype)}")
            elif etype.shape != (MAX_ENTITIES,):
                issues.append(f"entity_type shape {etype.shape} != ({MAX_ENTITIES},)")
            elif not np.issubdtype(etype.dtype, np.integer):
                issues.append(f"entity_type dtype {etype.dtype} not integer")
            elif np.any((etype < 0) | (etype >= NUM_ENTITY_TYPES)):
                issues.append(f"entity_type values out of [0, {NUM_ENTITY_TYPES-1}]")

            if not isinstance(mask, np.ndarray):
                issues.append(f"mask must be np.ndarray, got {type(mask)}")
            elif mask.shape != (MAX_ENTITIES,):
                issues.append(f"mask shape {mask.shape} != ({MAX_ENTITIES},)")
            elif not np.all((mask == 0) | (mask == 1)):
                issues.append("mask values must be 0 or 1")
            else:
                # Padding slots (mask=0) should have entity_type == 0 (pad)
                if isinstance(etype, np.ndarray):
                    pad_slots = mask == 0
                    if np.any(etype[pad_slots] != 0):
                        issues.append(
                            "Warning: entity_type != 0 in padding slots (mask=0). "
                            "EnemyAttentionModule masks these out, but it is unusual."
                        )

    return issues


# ---------------------------------------------------------------------------
# 2. Vitals normalisation (NumPy reference — mirrors VitalsEncoder.normalize_vitals)
# ---------------------------------------------------------------------------

def normalize_vitals_numpy(vitals: np.ndarray) -> np.ndarray:
    """Normalise a (17,) or (..., 17) vitals array to neural-network friendly scale.

    Mirrors ``network.encoder.VitalsEncoder.normalize_vitals()`` exactly so the
    two implementations produce the same values. Use this in preprocessing
    pipelines or for inspecting raw vs normalised vitals side by side.

    Parameters
    ----------
    vitals : np.ndarray
        Raw float32 vitals vector of shape (17,) or (..., 17).

    Returns
    -------
    np.ndarray
        Normalised copy, same shape, float32. Original is not modified.

    Normalisation table
    -------------------
    [0:2]   health, armor       / 100.0        → typical [0, 2]
    [2:3]   selected_weapon     / 7.0          → [0, 1]
    [3:8]   ammo (5 fields)     log1p / 4.0    → ≈[0, 1.6] for max ammo
    [8:14]  key flags           unchanged      → {0, 1}
    [14:17] ego velocity (3)    / 20.0         → running ≈ 0.6
    """
    v = vitals.astype(np.float32, copy=True)
    v[..., 0:2]   = v[..., 0:2]  * _HEALTH_ARMOR_SCALE
    v[..., 2:3]   = v[..., 2:3]  * _WEAPON_SCALE
    ammo = np.maximum(v[..., 3:8], 0.0)
    v[..., 3:8]   = np.log1p(ammo) * _AMMO_LOG_SCALE
    # indices 8:14 are key flags — already in {0, 1}, no scaling needed
    v[..., 14:17] = v[..., 14:17] * _VELOCITY_SCALE
    return v


# ---------------------------------------------------------------------------
# 3. Entity feature normalisation  (Audit Finding 3)
# ---------------------------------------------------------------------------

def normalize_entity_features(features: np.ndarray) -> np.ndarray:
    """Normalise raw entity features before they enter EnemyAttentionModule.

    Motivation (audit Finding 3)
    ----------------------------
    The raw entity feature vector has mixed scales:
      col 0  distance     [0, 4.0]  — entity at max 4096 units / 1024 = 4.0
      col 1  sin_rel_angle[-1, 1]   — already well-scaled
      col 2  cos_rel_angle[-1, 1]   — already well-scaled
      col 3  rel_vel_fwd  unbounded — fast projectiles can exceed ±50 u/tic
      col 4  rel_vel_right unbounded

    Large relative velocities dominate the first linear layer of the token encoder.
    Normalising here improves gradient flow in the early training phase.

    Parameters
    ----------
    features : np.ndarray
        Shape (32, 5) or (..., 32, 5), float32.
        The raw ``visible_entities["features"]`` array from the obs dict.

    Returns
    -------
    np.ndarray
        Normalised copy, same shape, float32. Original is not modified.

    Normalisation applied
    ---------------------
    col 0  distance    / 4.0            → [0, 1.0]
    col 1  sin_angle   unchanged        → [-1, 1]
    col 2  cos_angle   unchanged        → [-1, 1]
    col 3  rel_vel_fwd  clip ±20 / 20  → [-1, 1]
    col 4  rel_vel_right clip ±20 / 20 → [-1, 1]

    Important
    ---------
    Call this BEFORE inserting features into the obs dict that you pass to the
    network. The network's EnemyAttentionModule receives features as-is.
    Padding slots (mask=0) contain zeros and are unaffected.
    """
    f = features.astype(np.float32, copy=True)
    # Distance: [0, 4.0] → [0, 1.0]
    f[..., 0]   = f[..., 0] * _ENT_DIST_SCALE
    # sin / cos: already [-1, 1] — no change needed
    # Relative velocities: clip to ±20 u/tic then scale to [-1, 1]
    f[..., 3:5] = np.clip(f[..., 3:5], -_ENT_VEL_CLIP, _ENT_VEL_CLIP) * _ENT_VEL_SCALE
    return f


# ---------------------------------------------------------------------------
# 4. NumPy → PyTorch tensor conversion
# ---------------------------------------------------------------------------

def obs_to_device_tensors(obs: dict, device: str = "cpu") -> dict:
    """Convert a raw (numpy) obs dict to PyTorch tensors on `device`.

    Only the keys that DoomNetwork._extract_inputs() reads are converted:
    - vitals              → float32 tensor
    - rays.distance       → float32 tensor
    - rays.hit_type       → int64   tensor
    - visible_entities.features   → float32 tensor
    - visible_entities.entity_type→ int64   tensor
    - visible_entities.mask       → float32 tensor (attention module expects float)
    - visual_embedding (if present) → float32 tensor

    rgb and depth are NOT converted because the network ignores them by default.

    Parameters
    ----------
    obs : dict
        Raw observation dict from DoomEnv.
    device : str
        PyTorch device string, e.g. "cpu", "cuda", "mps".

    Returns
    -------
    dict
        New dict with the same structure; numpy arrays replaced by tensors.
        rgb and depth keys are carried over as-is (numpy, unchanged).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for obs_to_device_tensors()")

    def _t(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype, device=device)

    out = dict(obs)  # shallow copy — we will replace individual keys
    out["vitals"] = _t(obs["vitals"], torch.float32)
    out["rays"] = {
        "distance": _t(obs["rays"]["distance"], torch.float32),
        "hit_type":  _t(obs["rays"]["hit_type"],  torch.int64),
    }
    out["visible_entities"] = {
        "features":    _t(obs["visible_entities"]["features"],    torch.float32),
        "entity_type": _t(obs["visible_entities"]["entity_type"], torch.int64),
        "mask":        _t(obs["visible_entities"]["mask"],        torch.float32),
    }
    if "visual_embedding" in obs and obs["visual_embedding"] is not None:
        ve = obs["visual_embedding"]
        if isinstance(ve, np.ndarray):
            out["visual_embedding"] = _t(ve, torch.float32)
        # If it's already a tensor, leave it (or move device):
        elif hasattr(ve, "to"):
            out["visual_embedding"] = ve.to(device=device, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# 5. Main entry point: process_observation()
# ---------------------------------------------------------------------------

def process_observation(
    obs: dict,
    device: str = "cpu",
    *,
    normalize_entities: bool = True,
    visual_encoder=None,
) -> dict:
    """Enrich the raw observation dict for DoomNetwork.

    This is the single function called from ``submission/agent.py``:

        enriched_obs = process_observation(raw_obs, device=self.device,
                                           visual_encoder=self.visual_encoder)
        action_np, self.hx = self.model.get_action(enriched_obs, hx=self.hx)

    What it does
    ------------
    1. Optionally normalises entity features (audit Finding 3).
       When ``normalize_entities=True``, features[:,3:5] are clipped and scaled
       before the dict reaches EnemyAttentionModule.
    2. Optionally runs a visual encoder CNN on rgb/depth and injects the result
       as obs["visual_embedding"] so FeatureFusion can consume it.
    3. Returns a *new* dict — the original obs is never modified.

    Parameters
    ----------
    obs : dict
        Raw observation from DoomEnv.step() or env.reset().
    device : str
        PyTorch device (passed to visual_encoder if provided).
    normalize_entities : bool
        Apply normalize_entity_features() to visible_entities.features.
        Defaults to True. Set False only if your network already normalises
        entity inputs internally.
    visual_encoder : callable | None
        Optional visual encoder module. If provided, it must be callable as:
            visual_embedding = visual_encoder(obs["rgb"], obs["depth"], device)
        It must return a numpy array or tensor of shape (D,) float32.
        visual_feature_dim must be set to D in NetworkConfig.

    Returns
    -------
    dict
        Shallow copy of obs, with:
        - visible_entities["features"] optionally normalised (new array, not in-place)
        - "visual_embedding" key added if visual_encoder is not None
    """
    # Shallow copy — do not mutate the caller's dict
    enriched = dict(obs)

    # Normalise entity features if requested
    if normalize_entities:
        enriched["visible_entities"] = dict(obs["visible_entities"])
        enriched["visible_entities"]["features"] = normalize_entity_features(
            obs["visible_entities"]["features"]
        )

    # Run optional visual encoder
    if visual_encoder is not None:
        enriched["visual_embedding"] = visual_encoder(
            obs["rgb"], obs["depth"], device
        )

    return enriched
