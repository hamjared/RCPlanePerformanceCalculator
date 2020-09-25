from unittest import TestCase

from RCPlanePerformance.Units import units
from RCPlanePerformance.XFoilDragPolar import XFoilDragPolar


class TestQuadraticDragPolar(TestCase):
    def test_get_profile_drag(self):
        dragPolar = XFoilDragPolar('23012', re=5e6)
        self.assertAlmostEqual(dragPolar.get_profile_drag(4*units.degree), 0.006, 3)
