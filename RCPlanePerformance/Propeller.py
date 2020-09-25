import math

import pint

from RCPlanePerformance import AtmosphericModel
from RCPlanePerformance.IPropeller import IPropeller
from RCPlanePerformance.Units import units


class Propeller(IPropeller):

    def __init__(self, diameter: pint.unit, pitch: pint.unit, ct_function):
        self.diameter = diameter
        self.pitch = pitch
        self.ct_function = ct_function

    def calculate_thrust(self, angular_velocity: pint.unit, aircraft_velocity: pint.unit, altitude: pint.unit) -> pint.unit:
        v = aircraft_velocity.to('m/s').magnitude
        n = angular_velocity.to('rpm').magnitude/60
        d = self.diameter.to('m').magnitude
        J = v/(n*d)
        c_t = self.ct_function(J)
        rho = AtmosphericModel.get_air_density(altitude).to('kg/m**3').magnitude
        thrust = c_t * rho * n**2 * d**4 * units.newton
        return thrust


    def calculate_power(self, angular_velocity: pint.unit, aircraft_velocity: pint.unit) -> pint.unit:
        pass


if __name__ == '__main__':
    propeller = Propeller(
        diameter=10 * units.inch,
        pitch=7 * units.inch,
        ct_function=lambda J: -.1118 * J - 0.0255 * J + 0.144 if (J < 0.4) else -.201 * J + 0.1659

    )
    print(propeller.calculate_thrust(9200*units.rpm, 30*units.mph, 4000 * units.ft))
