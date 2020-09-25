from unittest import TestCase
import numpy as np
from RCPlanePerformance.QuadraticDragPolar import QuadraticDragPolar
from RCPlanePerformance.SimpleTaperedWing import SimpleTaperedWing
from RCPlanePerformance.Units import units

aoa_list = np.array([-4, -2, 0, 2, 4, 6]) * units.degree
c_d_list = np.array([.00702, .00687, .0063, .00547, .00559, .00619]) * units.dimensionless


class TestSimpleTaperedWing(TestCase):

    def test_constructor(self):
        wing = SimpleTaperedWing(span=3 * units.meter,
                                 root_chord=3 * units.meter,
                                 tip_chord=3 * units.meter,
                                 infinite_wing_lift_slope=1 / units.degree,
                                 aoa_zero_lift=0 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list, c_d_list),
                                 ac_to_cg=5 * units.inch)

    def test_get_drag(self):
        wing = SimpleTaperedWing(span=10 * units.meter,
                                 root_chord=1 * units.meter,
                                 tip_chord=1 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)
        self.assertAlmostEqual(wing.get_drag_coefficient(4 * units.degree), 0.0138, 3)

    def test_get_lift_coefficient(self):
        wing = SimpleTaperedWing(span=10 * units.meter,
                                 root_chord=1 * units.meter,
                                 tip_chord=1 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)

        self.assertAlmostEqual(wing.get_lift_coefficient(4*units.degree), 0.484, 3)

    def test_wing_area(self):
        wing = SimpleTaperedWing(span=1 * units.meter,
                                 root_chord=0.3 * units.meter,
                                 tip_chord=0.3 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)
        expected = 0.3 * units.meter * units.meter
        self.assertEqual(wing.get_wing_area(), expected)
        wing = SimpleTaperedWing(span=1 * units.meter,
                                 root_chord=0.4 * units.meter,
                                 tip_chord=0.2 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)
        expected = (0.4 + 0.2) / 2 * units.meter * units.meter
        self.assertEqual(wing.get_wing_area(), expected)

    def test_aspect_ratio(self):
        wing = SimpleTaperedWing(span=1 * units.meter,
                                 root_chord=0.3 * units.meter,
                                 tip_chord=0.3 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)
        expected = 1 * 1 / (1 * 0.3) * units.dimensionless
        self.assertEqual(wing.aspect_ratio(), expected)

    def test_lift_slope(self):
        wing = SimpleTaperedWing(span=10 * units.meter,
                                 root_chord=1 * units.meter,
                                 tip_chord=1 * units.meter,
                                 infinite_wing_lift_slope=0.106 / units.degree,
                                 aoa_zero_lift=-1.5 * units.degree,
                                 drag_polar=QuadraticDragPolar(aoa_list=aoa_list, c_d_list=c_d_list),
                                 ac_to_cg=5 * units.inch,
                                 e=0.95)
        self.assertAlmostEqual(wing.get_wing_lift_slope().to('1/degree').magnitude, 0.088, places=2)
