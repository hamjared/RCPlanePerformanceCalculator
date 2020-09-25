from unittest import TestCase

from RCPlanePerformance.Propeller import Propeller
from RCPlanePerformance.Units import units


class TestSimplePropeller(TestCase):
    def test_calculate_thrust(self):
        prop = Propeller(10 * units.inch, 6 * units.inch)

        self.assertAlmostEqual(prop.calculate_thrust(10350 * units.rpm, 30 * units.mph), 7.8 * units.newton, 0)
