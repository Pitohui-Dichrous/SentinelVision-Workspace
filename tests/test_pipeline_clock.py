from __future__ import annotations

import unittest

from safety_pipeline.clock import PipelineClock


class PipelineClockTests(unittest.TestCase):
    def test_live_source_uses_supplied_monotonic_time(self):
        clock = PipelineClock()
        self.assertEqual(clock.next_timestamp("rtsp", 123.5, pts_ms=0.0, fps=25.0), 123.5)

    def test_valid_file_pts_is_used(self):
        clock = PipelineClock()
        values = [
            clock.next_timestamp("file", 100.0, pts_ms=0.0, fps=25.0),
            clock.next_timestamp("file", 200.0, pts_ms=40.0, fps=25.0),
            clock.next_timestamp("file", 300.0, pts_ms=80.0, fps=25.0),
        ]
        self.assertEqual(values, [0.0, 0.04, 0.08])

    def test_constant_zero_pts_falls_back_to_frame_time(self):
        clock = PipelineClock()
        values = [
            clock.next_timestamp("file", 100.0 + index, pts_ms=0.0, fps=25.0)
            for index in range(4)
        ]
        self.assertEqual(values, [0.0, 0.04, 0.08, 0.12])

    def test_nan_and_backwards_pts_never_stop_or_reverse_clock(self):
        clock = PipelineClock()
        values = [
            clock.next_timestamp("file", 1.0, pts_ms=0.0, fps=20.0),
            clock.next_timestamp("file", 2.0, pts_ms=50.0, fps=20.0),
            clock.next_timestamp("file", 3.0, pts_ms=float("nan"), fps=20.0),
            clock.next_timestamp("file", 4.0, pts_ms=20.0, fps=20.0),
        ]
        for actual, expected in zip(values, (0.0, 0.05, 0.1, 0.15)):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_fps_uses_deterministic_default(self):
        clock = PipelineClock(fallback_fps=10.0)
        first = clock.next_timestamp("file", 1.0, pts_ms=0.0, fps=0.0)
        second = clock.next_timestamp("file", 99.0, pts_ms=0.0, fps=float("nan"))
        self.assertEqual((first, second), (0.0, 0.1))

    def test_fallback_uses_decoded_frame_index_when_inference_skips_frames(self):
        clock = PipelineClock()
        values = [
            clock.next_timestamp(
                "file", 100.0 + index, pts_ms=0.0, fps=25.0, frame_index=index
            )
            for index in (0, 2, 4)
        ]
        for actual, expected in zip(values, (0.0, 0.08, 0.16)):
            self.assertAlmostEqual(actual, expected)

    def test_valid_variable_frame_rate_pts_is_not_clamped_to_nominal_fps(self):
        clock = PipelineClock()
        values = [
            clock.next_timestamp("file", 1.0, pts_ms=0.0, fps=25.0, frame_index=0),
            clock.next_timestamp("file", 2.0, pts_ms=20.0, fps=25.0, frame_index=1),
            clock.next_timestamp("file", 3.0, pts_ms=60.0, fps=25.0, frame_index=2),
        ]
        for actual, expected in zip(values, (0.0, 0.02, 0.06)):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_fallback_fps_is_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PipelineClock(fallback_fps=value)


if __name__ == "__main__":
    unittest.main()
