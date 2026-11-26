"""ObservationInspector — developer tool for visualising DOOM observations.

This module has zero runtime dependencies beyond NumPy and the standard library.
It is designed to run on any machine, with or without torch/vizdoom/gymnasium,
so you can inspect observations from saved files or from a live game.

Usage examples
--------------
# From a live game step:
    from perception import ObservationInspector
    inspector = ObservationInspector()
    env = make_env("maps/two_rooms.wad")
    obs, _ = env.reset()
    inspector.print_summary(obs)
    inspector.print_rays(obs)
    inspector.print_entities(obs)
    inspector.print_vitals(obs)
    inspector.save_text_report(obs, "step_000.txt")

# From the examples/ script:
    python examples/inspect_observation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import numpy as np

# ---------------------------------------------------------------------------
# Constants (inlined so this file works without doom_env installed)
# ---------------------------------------------------------------------------

_RAY_HIT_NAMES = ("none", "wall", "door", "locked", "enemy", "item")
_ENTITY_TYPE_NAMES = (
    "pad", "enemy_melee", "enemy_hitscan", "enemy_projectile",
    "projectile", "health", "armor", "ammo", "weapon", "key", "barrel", "other",
)
_VITALS_NAMES = (
    "health", "armor", "selected_weapon", "sel_weapon_ammo",
    "ammo_bullets", "ammo_shells", "ammo_rockets", "ammo_cells",
    "key_blue_card", "key_yellow_card", "key_red_card",
    "key_blue_skull", "key_yellow_skull", "key_red_skull",
    "vel_forward", "vel_right", "vel_up",
)

NUM_RAYS       = 64
MAX_ENTITIES   = 32
RAY_RANGE      = 1024.0
ENTITY_DIST_MAX = 4.0   # = 4096 / 1024

# Hit-type characters for the ASCII ring diagram
_HIT_CHAR = {0: ".", 1: "W", 2: "D", 3: "L", 4: "E", 5: "I"}


class ObservationInspector:
    """Pretty-print DOOM observations for debugging and development.

    All methods accept the raw obs dict exactly as returned by DoomEnv and
    write to stdout (or any file-like object via the `file` kwarg).
    """

    # ------------------------------------------------------------------
    def print_summary(self, obs: dict, file: TextIO = sys.stdout) -> None:
        """One-line summary of every field — shapes, dtypes, ranges."""
        w = file.write
        w("=" * 70 + "\n")
        w("  DOOM OBSERVATION SUMMARY\n")
        w("=" * 70 + "\n")
        self._check_key(obs, "rgb",   (120, 160, 3), np.uint8,   file)
        self._check_key(obs, "depth", (120, 160),    np.uint8,   file)
        self._check_key(obs, "vitals",(17,),         np.float32, file)

        # Rays
        if "rays" in obs:
            d = obs["rays"].get("distance")
            h = obs["rays"].get("hit_type")
            d_ok = isinstance(d, np.ndarray) and d.shape == (64,) and d.dtype == np.float32
            h_ok = isinstance(h, np.ndarray) and h.shape == (64,)
            w(f"  rays.distance     (64,) float32  range=[{d.min():.3f}, {d.max():.3f}]"
              f"  {'OK' if d_ok else 'SHAPE/DTYPE ERROR'}\n")
            hit_counts = {name: int((h == i).sum()) for i, name in enumerate(_RAY_HIT_NAMES)}
            w(f"  rays.hit_type     (64,)           counts={hit_counts}\n")
        if "visible_entities" in obs:
            ve = obs["visible_entities"]
            mask  = ve.get("mask",  np.zeros(32, np.int8))
            feats = ve.get("features", np.zeros((32, 5), np.float32))
            etype = ve.get("entity_type", np.zeros(32, np.int64))
            n_real = int((mask == 1).sum())
            w(f"  visible_entities  {n_real} real / {MAX_ENTITIES} total slots\n")
            if n_real:
                dist_col = feats[mask == 1, 0] * RAY_RANGE
                w(f"    dist range       [{dist_col.min():.0f}, {dist_col.max():.0f}] map units\n")
                type_names = [_ENTITY_TYPE_NAMES[int(t)] for t in etype[mask == 1]]
                w(f"    types            {type_names}\n")
        w("=" * 70 + "\n")

    # ------------------------------------------------------------------
    def print_vitals(self, obs: dict, file: TextIO = sys.stdout) -> None:
        """Print each vitals field with name, raw value and normalised value."""
        vitals = obs.get("vitals")
        if vitals is None:
            file.write("vitals: NOT PRESENT\n")
            return
        file.write("\n  VITALS  (raw → normalised)\n")
        file.write("  " + "-" * 50 + "\n")
        norm = _normalize_vitals_simple(vitals)
        for i, name in enumerate(_VITALS_NAMES):
            bar = _mini_bar(norm[i], -2.0, 2.0)
            file.write(f"  [{i:2d}] {name:<24s}  {vitals[i]:>10.3f}  →  {norm[i]:>7.3f}  {bar}\n")

    # ------------------------------------------------------------------
    def print_rays(self, obs: dict, file: TextIO = sys.stdout) -> None:
        """Print a 64-ray table and an ASCII overhead ring diagram."""
        rays = obs.get("rays", {})
        distance = rays.get("distance", np.ones(NUM_RAYS, np.float32))
        hit_type = rays.get("hit_type", np.zeros(NUM_RAYS, np.int64))

        file.write("\n  RAYS  (index | angle | hit | dist_norm | dist_units)\n")
        file.write("  " + "-" * 60 + "\n")
        for i in range(NUM_RAYS):
            angle = i * (360.0 / NUM_RAYS)
            d_norm = float(distance[i])
            d_units = d_norm * RAY_RANGE
            hname = _RAY_HIT_NAMES[int(hit_type[i])]
            bar = _mini_bar(d_norm, 0.0, 1.0, width=20, fill="─", empty="·")
            file.write(
                f"  [{i:2d}] {angle:6.1f}°  {hname:<8s}  {d_norm:.3f}  "
                f"({d_units:6.1f} u)  {bar}\n"
            )
        file.write("\n")
        self._print_ray_ring(distance, hit_type, file)

    # ------------------------------------------------------------------
    def print_entities(self, obs: dict, file: TextIO = sys.stdout) -> None:
        """Print a table of visible entities with all 5 feature columns."""
        ve = obs.get("visible_entities", {})
        feats = ve.get("features",    np.zeros((MAX_ENTITIES, 5), np.float32))
        etype = ve.get("entity_type", np.zeros(MAX_ENTITIES, np.int64))
        mask  = ve.get("mask",        np.zeros(MAX_ENTITIES, np.int8))

        n_real = int((mask == 1).sum())
        file.write(f"\n  VISIBLE ENTITIES  ({n_real} / {MAX_ENTITIES} slots occupied)\n")
        file.write("  " + "-" * 80 + "\n")
        file.write(
            f"  {'#':>2}  {'type':<20s}  {'dist(norm)':>10}  {'dist(u)':>8}  "
            f"{'sin_ang':>7}  {'cos_ang':>7}  {'vfwd':>7}  {'vright':>7}\n"
        )
        file.write("  " + "-" * 80 + "\n")
        for i in range(MAX_ENTITIES):
            if mask[i] == 0:
                continue
            tname = _ENTITY_TYPE_NAMES[int(etype[i])]
            d_norm = feats[i, 0]
            d_u    = d_norm * RAY_RANGE
            sin_a  = feats[i, 1]
            cos_a  = feats[i, 2]
            vf     = feats[i, 3]
            vr     = feats[i, 4]
            angle_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))
            file.write(
                f"  [{i:2d}] {tname:<20s}  {d_norm:>10.4f}  {d_u:>8.1f}  "
                f"{sin_a:>7.3f}  {cos_a:>7.3f}  {vf:>7.2f}  {vr:>7.2f}"
                f"  ({angle_deg:+.1f}°)\n"
            )
        file.write("\n  Legend: dist_norm = distance/1024; dist(u) = distance in map units\n")
        file.write("          angle: 0°=ahead, +left, -right\n")

    # ------------------------------------------------------------------
    def print_full_report(self, obs: dict, file: TextIO = sys.stdout) -> None:
        """Print all sections in sequence."""
        self.print_summary(obs, file)
        self.print_vitals(obs, file)
        self.print_rays(obs, file)
        self.print_entities(obs, file)

    # ------------------------------------------------------------------
    def save_text_report(self, obs: dict, path: str | Path) -> None:
        """Save the full report to a plain text file."""
        with open(path, "w", encoding="utf-8") as fh:
            self.print_full_report(obs, fh)
        print(f"[ObservationInspector] Report saved to {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_key(obs, key, expected_shape, expected_dtype, file):
        arr = obs.get(key)
        if arr is None:
            file.write(f"  {key:<18s}  MISSING\n")
            return
        ok   = isinstance(arr, np.ndarray) and arr.shape == expected_shape and arr.dtype == expected_dtype
        rng  = f"[{arr.min()}, {arr.max()}]" if arr.size else "empty"
        file.write(
            f"  {key:<18s}  {str(arr.shape):<16s}  {str(arr.dtype):<8s}  "
            f"range={rng:<20s}  {'OK' if ok else 'SHAPE/DTYPE ERROR'}\n"
        )

    @staticmethod
    def _print_ray_ring(distance: np.ndarray, hit_type: np.ndarray, file: TextIO) -> None:
        """Print a compact ASCII overhead-view ring of 64 rays.

        Legend: W=wall  D=door  L=locked  E=enemy  I=item  .=clear
        A circle of 64 characters; the top of the ring is Ray 0 (forward).
        """
        ring_size = 32          # characters around the circle
        step = NUM_RAYS // ring_size
        ring = []
        for j in range(ring_size):
            ray_idx = (j * step) % NUM_RAYS
            d = distance[ray_idx]
            h = int(hit_type[ray_idx])
            ch = _HIT_CHAR.get(h, "?")
            ring.append(ch if d < 0.95 else "·")

        file.write("  Ray ring (top=forward, clockwise, 32 of 64 rays):\n")
        # Print as a 3-line block: top arc, L/R, bottom arc
        top    = ring[:8]
        r_side = ring[8:16]
        bot    = ring[16:24][::-1]
        l_side = ring[24:32][::-1]
        file.write("    " + "".join(top) + "\n")
        for i in range(8):
            file.write(f"  {l_side[i]}  {'P' if i == 3 else ' ' * 10}  {r_side[i]}\n")
        file.write("    " + "".join(bot) + "\n")
        file.write("  Legend: W=wall D=door L=locked E=enemy I=item ·=clear/open\n\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_vitals_simple(v: np.ndarray) -> np.ndarray:
    """Mirror of VitalsEncoder.normalize_vitals() for display purposes."""
    out = v.copy().astype(np.float32)
    out[0:2]   *= (1.0 / 100.0)
    out[2:3]   *= (1.0 / 7.0)
    ammo = np.maximum(out[3:8], 0.0)
    out[3:8]    = np.log1p(ammo) / 4.0
    out[14:17] *= (1.0 / 20.0)
    return out


def _mini_bar(value: float, lo: float, hi: float, width: int = 15,
              fill: str = "█", empty: str = " ") -> str:
    """Render a horizontal bar proportional to `value` in [lo, hi]."""
    fraction = max(0.0, min(1.0, (value - lo) / max(hi - lo, 1e-9)))
    filled = round(fraction * width)
    return "[" + fill * filled + empty * (width - filled) + "]"
