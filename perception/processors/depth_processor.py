"""DepthProcessor — depth buffer perception from the DOOM screen.

Converts obs["depth"] (120×160 uint8) into a typed DepthFeatures dataclass
containing normalised arrays and compact 1-D depth profiles that can be used
as lightweight visual features without running a full CNN.

Observation contract (env/doom_env/spaces.py)
---------------------------------------------
obs["depth"]   (120, 160)  uint8
  The ViZDoom depth buffer is a **perspective-depth** (z-buffer) value
  encoded into uint8. It is NOT a linear distance map — nearby surfaces
  get lower values and far surfaces get higher values, but the mapping
  is non-linear (perspective-corrected). Do NOT use this as a metric
  distance without additional calibration.
  0   = very close surface (or entire buffer is 0 when render_buffers=False).
  255 = furthest depth the buffer encodes.
  When render_buffers=False the entire buffer is zeros — this is VALID
  and is signalled by is_zero_buffer=True.

Design rules
------------
- Pure NumPy; zero torch dependency.
- Never modifies the caller's obs dict.
- is_zero_buffer=True is a clean no-op, not an error.
- All derived arrays are freshly allocated.
- Must complete in well under 1 ms on a single CPU thread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Screen geometry constants (from doom_env.spaces)
# ---------------------------------------------------------------------------

SCREEN_HEIGHT = 120
SCREEN_WIDTH  = 160

# Canonical row indices for the 1-D depth profiles
_HORIZON_ROW = 60    # vertical centre — sky/wall boundary zone
_FLOOR_ROW   = 100   # lower third   — floor visibility zone

# Normalisation divisor: uint8 → [0.0, 1.0]
_DEPTH_SCALE = 1.0 / 255.0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthFeatures:
    """All derived depth buffer features.

    Fields
    ------
    raw : (120, 160) uint8
        Unchanged copy of obs["depth"].

    normalized : (120, 160) float32
        raw / 255.0  →  values in [0.0, 1.0].
        0 = closest surface, 1 = furthest visible surface.
        All zeros when is_zero_buffer is True.

    is_zero_buffer : bool
        True when render_buffers=False (all pixels are 0).
        Downstream code should skip visual feature extraction when True.

    horizon_strip : (160,) float32
        Normalised depth values along row 60 (vertical centre).
        Captures the depth profile at roughly the horizon line.

    floor_strip : (160,) float32
        Normalised depth values along row 100 (lower third of screen).
        Captures nearby floor/obstacle depth.

    col_min : (160,) float32
        Per-column minimum normalised depth across all rows.
        Represents the closest surface visible in each screen column.

    col_mean : (160,) float32
        Per-column mean normalised depth across all rows.
        Useful for distinguishing open corridors from filled columns.

    compact_features : (480,) float32
        Concatenation of [horizon_strip | floor_strip | col_min].
        A compact 480-dim visual summary usable as a linear feature vector
        without running a CNN.  All zeros when is_zero_buffer is True.
    """

    raw:               np.ndarray   # (120, 160) uint8
    normalized:        np.ndarray   # (120, 160) float32
    is_zero_buffer:    bool
    horizon_strip:     np.ndarray   # (160,)     float32
    floor_strip:       np.ndarray   # (160,)     float32
    col_min:           np.ndarray   # (160,)     float32
    col_mean:          np.ndarray   # (160,)     float32
    compact_features:  np.ndarray   # (480,)     float32


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class DepthProcessor:
    """Stateless processor for the (120, 160) depth buffer.

    Usage
    -----
    processor = DepthProcessor()
    features  = processor.process(obs)

    Parameters
    ----------
    horizon_row : int
        Row index for the horizon depth strip. Default 60 (vertical centre).
    floor_row : int
        Row index for the floor depth strip. Default 100 (lower third).
    """

    def __init__(
        self,
        horizon_row: int = _HORIZON_ROW,
        floor_row:   int = _FLOOR_ROW,
    ) -> None:
        self._horizon_row = int(horizon_row)
        self._floor_row   = int(floor_row)

    # ------------------------------------------------------------------
    def process(self, obs: dict) -> DepthFeatures:
        """Process obs["depth"] into a fully-validated DepthFeatures object.

        Parameters
        ----------
        obs : dict
            Raw observation dict from DoomEnv.

        Returns
        -------
        DepthFeatures
            Immutable dataclass containing normalised arrays.
            If "depth" is missing or malformed, or if render_buffers=False
            produced an all-zero buffer, all derived arrays are zero and
            is_zero_buffer=True.  Callers should check is_zero_buffer before
            using depth features as a visual input signal.
        """
        raw = self._extract_safe(obs)

        # Detect zero-buffer (render_buffers=False or missing depth) first.
        # When True all derived arrays are already zeros — no further work needed.
        is_zero_buffer = not np.any(raw)

        # Normalise: uint8 [0, 255] → float32 [0.0, 1.0].
        # NOTE: ViZDoom depth is a perspective z-buffer, not linear distance.
        # Treat these values as a relative-depth signal only.
        normalized = raw.astype(np.float32) * _DEPTH_SCALE  # (120, 160)

        # --- 1-D depth profiles ----------------------------------------
        # Clamp row indices defensively in case of custom screen sizes.
        h_row = min(self._horizon_row, SCREEN_HEIGHT - 1)
        f_row = min(self._floor_row,   SCREEN_HEIGHT - 1)

        # When is_zero_buffer these slices are already all-zero — correct.
        horizon_strip = normalized[h_row, :].copy()   # (160,)
        floor_strip   = normalized[f_row, :].copy()   # (160,)

        # --- Per-column statistics (vectorised over rows) ----------------
        col_min  = normalized.min(axis=0)   # (160,) float32
        col_mean = normalized.mean(axis=0)  # (160,) float32

        # --- Compact feature vector -------------------------------------
        # [horizon_strip | floor_strip | col_min] = 160+160+160 = 480 dims.
        # All-zero when is_zero_buffer — do NOT pass to a visual encoder
        # as if it represents valid depth data.
        compact_features = np.concatenate(
            [horizon_strip, floor_strip, col_min]
        ).astype(np.float32)   # (480,)

        return DepthFeatures(
            raw=raw,
            normalized=normalized,
            is_zero_buffer=is_zero_buffer,
            horizon_strip=horizon_strip,
            floor_strip=floor_strip,
            col_min=col_min,
            col_mean=col_mean,
            compact_features=compact_features,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_safe(obs: dict) -> np.ndarray:
        """Return a (SCREEN_HEIGHT, SCREEN_WIDTH) uint8 depth array.

        Falls back to all-zeros on any malformation — identical to what
        the env returns when render_buffers=False.
        """
        raw = obs.get("depth", None)
        if (
            raw is None
            or not isinstance(raw, np.ndarray)
            or raw.shape != (SCREEN_HEIGHT, SCREEN_WIDTH)
        ):
            return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)

        # Ensure correct dtype without modifying caller's array
        if raw.dtype != np.uint8:
            return raw.astype(np.uint8)
        return raw.copy()
