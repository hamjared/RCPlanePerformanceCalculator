from unittest import TestCase

from RCPlanePerformance.AtmosphericModel import get_air_density
from RCPlanePerformance.Units import units


class Test(TestCase):
    def test_get_air_density(self):
        rho = get_air_density(0 * units.meter)
        expected = 1.225 * units.kg/(units.meter**3)
        self.assertAlmostEqual(rho.to('kg/(m**3)').magnitude, expected.magnitude,2)

        rho = get_air_density(1000 * units.meter)
        expected = 1.1117 * units.kg / (units.meter ** 3)
        self.assertAlmostEqual(rho.to('kg/(m**3)').magnitude, expected.magnitude, 2)

        rho = get_air_density(1000 * units.feet)
        expected = 2.3081 * 10**-3 * units.slugs / (units.ft ** 3)
        self.assertAlmostEqual(rho.to('kg/(m**3)').magnitude, expected.to('kg/(m**3)').magnitude, 2)

