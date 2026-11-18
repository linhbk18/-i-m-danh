"""Regression test for the 2026-11-18 fix: Continuity's track-matching window
must survive REAL measured per-frame inference latency, or the 5-of-7
stability window (backend/attendance/stability.py) resets to zero even when
ArcFace recognizes the same in-class student correctly on every single call.

Before the fix, update()/bind() required consecutive YuNet preview ticks to
land within 3s of each other (snapshot()/valid() used 2.5s). Real single-face
RetinaFace+ArcFace latency on this CPU-only machine measured 1.6-2.4s (see
AI_ATTENDANCE_UPGRADE_REPORT.md and outputs/real_bench_fixed.json), and
multi-face up to ~5.9s -- tight enough that a single slightly slow inference
cycle silently started a new track id, wiping the accumulated stability
window (confirmed live with the real model: see conversation notes on
repro_auto_bug.py). The fix widened the window to 8s. These tests use
synthetic timestamps (no real sleeping) so they stay fast and deterministic;
they do not call ArcFace/RetinaFace and are not evidence of recognition
accuracy -- only of track continuity surviving realistic timing.

Run: .venv/Scripts/python.exe -m unittest tests.test_continuity_timing -v
"""
import unittest
import numpy as np
from backend.attendance.continuity import Continuity

FRAME = dict(width=640, height=480)


def face(x=100, y=100, w=120, h=140):
    return dict(box=dict(x=x, y=y, w=w, h=h))


def image():
    return np.random.default_rng(1).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


class ContinuityRealisticLatency(unittest.TestCase):
    def test_track_survives_4_second_gap_between_preview_ticks(self):
        """4s exceeds the OLD 3s window but is well within one real single-face
        inference cycle (~1.6-2.4s) plus the frontend's own re-schedule delay
        and normal jitter; it must not be treated as a different person."""
        c = Continuity()
        img = image()
        c.update([face()], FRAME, img, at=0.0)
        c.update([face()], FRAME, img, at=4.0)
        c.update([face()], FRAME, img, at=8.0)
        self.assertEqual(len(c.tracks), 1)

    def test_bind_and_valid_survive_measured_single_face_worst_case(self):
        c = Continuity()
        img = image()
        c.update([face()], FRAME, img, at=0.0)
        snap = c.snapshot(img, at=0.1)
        self.assertEqual(len(snap), 1)
        # 2.4s: measured worst-case single-face /api/recognize latency today.
        linked = c.bind([face()], FRAME, snap, now=2.4)
        self.assertIsNotNone(linked[0])
        self.assertTrue(c.valid(linked[0], now=2.4, max_gap=8))

    def test_multi_face_worst_case_latency_does_not_fragment_track(self):
        c = Continuity()
        img = image()
        c.update([face()], FRAME, img, at=0.0)
        # 5.9s: measured worst-case 6-face /api/recognize latency today.
        c.update([face()], FRAME, img, at=5.9)
        self.assertEqual(len(c.tracks), 1)

    def test_gap_past_8s_still_starts_a_new_track(self):
        """The window is wide, not infinite: a face genuinely gone for 20s
        must not silently inherit an old, possibly-wrong track identity."""
        c = Continuity()
        img = image()
        c.update([face()], FRAME, img, at=0.0)
        first = c.tracks[0]['id']
        c.update([face()], FRAME, img, at=20.0)
        self.assertNotEqual(c.tracks[0]['id'], first)


if __name__ == '__main__':
    unittest.main()
