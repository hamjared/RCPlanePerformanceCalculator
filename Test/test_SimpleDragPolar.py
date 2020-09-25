from unittest import TestCase

from RCPlanePerformance.QuadraticDragPolar import QuadraticDragPolar
from RCPlanePerformance.Units import units


class TestSimpleDragPolar(TestCase):
    def test_get_profile_drag(self):
        dp = QuadraticDragPolar(9 * 10 ** -5 * 1 / (units.degree ** 2), 0.0001 * 1 / units.degree, 0.01 * units.dimensionless)
        self.assertAlmostEqual(dp.get_profile_drag(2 * units.degree), 0.0108, 3)
        self.assertAlmostEqual(dp.get_profile_drag(6 * units.degree), 0.0140, 3)

