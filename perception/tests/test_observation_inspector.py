"""Unit tests for perception/observation_inspector.py.

Dependency-free: requires only NumPy and the standard library.
Tests verify that every print method runs without raising an exception and
that save_text_report creates a readable file with expected sections.

Run with:
    python -m pytest perception/tests/test_observation_inspector.py -v
    # or directly:
    python perception/tests/test_observation_inspector.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make repo root importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from perception.observation_inspector import ObservationInspector, _mini_bar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_RAYS     = 64
MAX_ENTITIES = 32
SCREEN_H     = 120
SCREEN_W     = 160
NUM_VITALS   = 17


def _make_obs(
    n_entities: int = 3,
    health: float = 100.0,
    has_doors: bool = False,
    has_enemies: bool = False,
) -> dict:
    """Create a minimal obs dict suitable for ObservationInspector tests."""
    rgb   = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    depth = np.zeros((SCREEN_H, SCREEN_W),    dtype=np.uint8)

    vitals = np.zeros(NUM_VITALS, dtype=np.float32)
    vitals[0] = health      # health
    vitals[1] = 50.0        # armor
    vitals[2] = 2.0         # weapon
    vitals[3] = 25.0        # sel_weapon_ammo
    vitals[4] = 100.0       # ammo_bullets
    vitals[14] = 10.0       # vel_forward
    vitals[15] = -3.0       # vel_right

    ray_dist = np.ones(NUM_RAYS, dtype=np.float32) * 0.9
    ray_hit  = np.zeros(NUM_RAYS, dtype=np.int64)   # all open

    if has_doors:
        ray_hit[5]  = 2   # door
        ray_hit[20] = 3   # locked
        ray_dist[5] = 0.4
        ray_dist[20] = 0.6

    if has_enemies:
        ray_hit[0] = 4    # enemy ray straight ahead
        ray_dist[0] = 0.3

    feats = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
    etype = np.zeros(MAX_ENTITIES, dtype=np.int64)
    mask  = np.zeros(MAX_ENTITIES, dtype=np.int8)

    for i in range(min(n_entities, MAX_ENTITIES)):
        feats[i, 0] = 0.5 + i * 0.1
        feats[i, 1] = 0.0
        feats[i, 2] = 1.0
        etype[i] = 1          # enemy_melee
        mask[i] = 1

    return {
        "rgb":   rgb,
        "depth": depth,
        "vitals": vitals,
        "rays":  {"distance": ray_dist, "hit_type": ray_hit},
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


INSPECTOR = ObservationInspector()   # stateless — safe to share


# ===========================================================================
# SECTION 1: print_summary
# ===========================================================================

class TestPrintSummary:

    def test_no_crash_valid_obs(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        output = buf.getvalue()
        assert len(output) > 0

    def test_contains_key_field_names(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        output = buf.getvalue()
        assert "rgb" in output
        assert "depth" in output
        assert "vitals" in output
        assert "rays" in output
        assert "visible_entities" in output

    def test_shape_is_reported(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        output = buf.getvalue()
        # RGB shape should appear
        assert "120" in output or "(120" in output

    def test_entity_count_shown(self):
        obs = _make_obs(n_entities=5)
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        output = buf.getvalue()
        assert "5" in output   # 5 real entities

    def test_zero_entities(self):
        obs = _make_obs(n_entities=0)
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        assert buf.getvalue()  # must not crash

    def test_all_zero_rgb_still_prints(self):
        obs = _make_obs()
        obs["rgb"][:] = 0
        buf = io.StringIO()
        INSPECTOR.print_summary(obs, file=buf)
        assert buf.getvalue()


# ===========================================================================
# SECTION 2: print_vitals
# ===========================================================================

class TestPrintVitals:

    def test_no_crash_valid_obs(self):
        obs = _make_obs(health=75.0)
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        assert buf.getvalue()

    def test_all_vitals_names_in_output(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        output = buf.getvalue()
        for name in ("health", "armor", "vel_forward", "ammo_bullets"):
            assert name in output, f"Expected vitals field '{name}' in output"

    def test_missing_vitals_does_not_crash(self):
        obs = _make_obs()
        del obs["vitals"]
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        assert "NOT PRESENT" in buf.getvalue()

    def test_health_100_shown_in_output(self):
        obs = _make_obs(health=100.0)
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        output = buf.getvalue()
        assert "100" in output

    def test_dead_player_health_zero(self):
        obs = _make_obs(health=0.0)
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        assert buf.getvalue()   # must not crash

    def test_all_17_indices_shown(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_vitals(obs, file=buf)
        output = buf.getvalue()
        # Every index [0]...[16] should appear
        for i in range(NUM_VITALS):
            assert f"[{i:2d}]" in output or f"[{i}]" in output or str(i) in output


# ===========================================================================
# SECTION 3: print_rays
# ===========================================================================

class TestPrintRays:

    def test_no_crash_valid_obs(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        assert buf.getvalue()

    def test_all_64_ray_indices_shown(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        output = buf.getvalue()
        # Spot-check a few indices
        for i in [0, 16, 32, 63]:
            assert f"[{i:2d}]" in output or f"[{i}]" in output

    def test_door_hit_type_in_output(self):
        obs = _make_obs(has_doors=True)
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        output = buf.getvalue()
        assert "door" in output.lower()

    def test_locked_door_in_output(self):
        obs = _make_obs(has_doors=True)
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        output = buf.getvalue()
        assert "locked" in output.lower()

    def test_enemy_ray_in_output(self):
        obs = _make_obs(has_enemies=True)
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        output = buf.getvalue()
        assert "enemy" in output.lower()

    def test_ascii_ring_diagram_present(self):
        obs = _make_obs()
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        output = buf.getvalue()
        # The ring diagram should contain the legend
        assert "W=wall" in output

    def test_missing_rays_does_not_crash(self):
        """If 'rays' key is absent, print_rays should use defaults gracefully."""
        obs = _make_obs()
        del obs["rays"]
        buf = io.StringIO()
        INSPECTOR.print_rays(obs, file=buf)
        assert buf.getvalue()


# ===========================================================================
# SECTION 4: print_entities
# ===========================================================================

class TestPrintEntities:

    def test_no_crash_valid_obs(self):
        obs = _make_obs(n_entities=3)
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        assert buf.getvalue()

    def test_no_crash_zero_entities(self):
        obs = _make_obs(n_entities=0)
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        assert buf.getvalue()

    def test_entity_type_name_in_output(self):
        obs = _make_obs(n_entities=1)
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        output = buf.getvalue()
        assert "enemy_melee" in output

    def test_slot_count_shown(self):
        obs = _make_obs(n_entities=5)
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        output = buf.getvalue()
        # Header should mention total slots
        assert str(MAX_ENTITIES) in output

    def test_angle_shown_in_output(self):
        obs = _make_obs(n_entities=1)
        # Set a 45° angle
        import math
        obs["visible_entities"]["features"][0, 1] = math.sin(math.radians(45))
        obs["visible_entities"]["features"][0, 2] = math.cos(math.radians(45))
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        output = buf.getvalue()
        assert "°" in output  # angle should be in output

    def test_legend_in_output(self):
        obs = _make_obs(n_entities=1)
        buf = io.StringIO()
        INSPECTOR.print_entities(obs, file=buf)
        output = buf.getvalue()
        assert "Legend" in output or "legend" in output.lower()


# ===========================================================================
# SECTION 5: print_full_report and save_text_report
# ===========================================================================

class TestFullReportAndSave:

    def test_print_full_report_no_crash(self):
        obs = _make_obs(n_entities=3, has_doors=True, has_enemies=True)
        buf = io.StringIO()
        INSPECTOR.print_full_report(obs, file=buf)
        output = buf.getvalue()
        assert len(output) > 100   # Must have produced substantial output

    def test_print_full_report_contains_all_sections(self):
        obs = _make_obs(n_entities=2)
        buf = io.StringIO()
        INSPECTOR.print_full_report(obs, file=buf)
        output = buf.getvalue()
        for section_marker in ("VITALS", "RAYS", "ENTITIES", "SUMMARY"):
            assert section_marker in output.upper(), \
                f"Expected section '{section_marker}' in full report"

    def test_save_text_report_creates_file(self, tmp_path):
        obs = _make_obs(n_entities=2)
        report_path = tmp_path / "test_report.txt"
        INSPECTOR.save_text_report(obs, report_path)
        assert report_path.exists(), "save_text_report did not create the file"
        assert report_path.stat().st_size > 0, "Report file is empty"

    def test_save_text_report_file_is_readable(self, tmp_path):
        obs = _make_obs(n_entities=1)
        report_path = tmp_path / "report.txt"
        INSPECTOR.save_text_report(obs, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "rgb" in content
        assert "vitals" in content

    def test_save_text_report_path_as_string(self, tmp_path):
        obs = _make_obs()
        report_path = str(tmp_path / "str_path_report.txt")
        INSPECTOR.save_text_report(obs, report_path)
        assert Path(report_path).exists()

    def test_full_report_does_not_modify_obs(self):
        obs = _make_obs(n_entities=3)
        orig_vitals = obs["vitals"].copy()
        buf = io.StringIO()
        INSPECTOR.print_full_report(obs, file=buf)
        np.testing.assert_array_equal(obs["vitals"], orig_vitals)


# ===========================================================================
# SECTION 6: _mini_bar helper
# ===========================================================================

class TestMiniBar:

    def test_full_bar(self):
        bar = _mini_bar(1.0, 0.0, 1.0, width=10)
        assert bar == "[██████████]"

    def test_empty_bar(self):
        bar = _mini_bar(0.0, 0.0, 1.0, width=10)
        assert bar == "[          ]"

    def test_half_bar(self):
        bar = _mini_bar(0.5, 0.0, 1.0, width=10)
        # 5 filled + 5 empty
        assert bar.count("█") == 5

    def test_clamps_above_range(self):
        bar = _mini_bar(2.0, 0.0, 1.0, width=8)
        assert bar == "[████████]"

    def test_clamps_below_range(self):
        bar = _mini_bar(-1.0, 0.0, 1.0, width=8)
        assert bar == "[        ]"

    def test_custom_fill_and_empty(self):
        bar = _mini_bar(1.0, 0.0, 1.0, width=5, fill="=", empty="-")
        assert bar == "[=====]"

    def test_always_starts_with_bracket(self):
        for v in [0.0, 0.5, 1.0]:
            bar = _mini_bar(v, 0.0, 1.0)
            assert bar.startswith("[")
            assert bar.endswith("]")


# ---------------------------------------------------------------------------
# Allow running directly: python perception/tests/test_observation_inspector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    import tempfile
    from pathlib import Path

    class _TmpPath:
        """Minimal tmp_path replacement for direct execution."""
        def __init__(self):
            self._d = tempfile.mkdtemp()
        def __truediv__(self, name):
            return Path(self._d) / name

    test_classes = [
        TestPrintSummary,
        TestPrintVitals,
        TestPrintRays,
        TestPrintEntities,
        TestFullReportAndSave,
        TestMiniBar,
    ]

    # Inject tmp_path where needed
    _NEEDS_TMP = {"test_save_text_report_creates_file",
                  "test_save_text_report_file_is_readable",
                  "test_save_text_report_path_as_string"}

    total = passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            total += 1
            try:
                fn = getattr(instance, name)
                if name in _NEEDS_TMP:
                    fn(_TmpPath())
                else:
                    fn()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
