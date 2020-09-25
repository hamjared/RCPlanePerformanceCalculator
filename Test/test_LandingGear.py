from unittest import TestCase

from RCPlanePerformance.LandingGear import LandingGear
from RCPlanePerformance.Units import units


class TestLandingGear(TestCase):
    def test_get_drag_coefficient(self):
        lg = LandingGear(diameter=3*units.inch, width=0.5*units.inch, reference_area=1 * units.inch**2)
        self.assertAlmostEqual(2*lg.calc_drag_coefficient(), 3.03 * units.dimensionless, 2)
