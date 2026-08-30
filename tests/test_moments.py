import unittest

from salsbury_md_analysis.moments import (
    CoordinateMoments,
    DisplacementCovariance,
    MomentError,
)


class MomentStateTests(unittest.TestCase):
    def test_coordinate_state_round_trip_and_merge_equal_direct_state(self):
        left = CoordinateMoments(1)
        right = CoordinateMoments(1)
        direct = CoordinateMoments(1)
        for value in (0.0, 2.0):
            left.update(((value, 0.0, 0.0),))
            direct.update(((value, 0.0, 0.0),))
        for value in (10.0, 14.0):
            right.update(((value, 0.0, 0.0),))
            direct.update(((value, 0.0, 0.0),))
        restored = CoordinateMoments.from_state(left.to_state())
        restored.merge(CoordinateMoments.from_state(right.to_state()))
        self.assertEqual(restored.count, direct.count)
        self.assertAlmostEqual(restored.mean_coordinate(0)[0], 6.5)
        self.assertAlmostEqual(restored.rmsf(0), direct.rmsf(0))

    def test_covariance_state_round_trip_and_merge_equal_direct_state(self):
        left = DisplacementCovariance(2)
        right = DisplacementCovariance(2)
        direct = DisplacementCovariance(2)
        frames = (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((4.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            ((7.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        )
        for frame in frames[:2]:
            left.update(frame)
            direct.update(frame)
        for frame in frames[2:]:
            right.update(frame)
            direct.update(frame)
        restored = DisplacementCovariance.from_state(left.to_state())
        restored.merge(DisplacementCovariance.from_state(right.to_state()))
        self.assertEqual(restored.count, direct.count)
        self.assertEqual(
            restored.correlation_matrix(1.0e-12),
            direct.correlation_matrix(1.0e-12),
        )

    def test_state_shape_validation_is_fail_closed(self):
        state = CoordinateMoments(2).to_state()
        state["mean"] = [[0.0, 0.0, 0.0]]
        with self.assertRaises(MomentError):
            CoordinateMoments.from_state(state)


if __name__ == "__main__":
    unittest.main()
