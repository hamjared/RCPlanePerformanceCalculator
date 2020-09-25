import math

import pint

from RCPlanePerformance.Units import units


def get_air_density(altitude: pint.unit) -> pint.unit:
    # from: https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
    h = altitude.to('m').magnitude
    temp = 0
    pressure = 0
    if h > 25000:
        pass
    elif h > 11000:
        temp = -55.46
        pressure = 22.65 * math.exp(1.73 - 0.00157 * h)
    else:
        temp = 15.04 - 0.00649 * h
        pressure = 101.29 * ((temp+273.1)/288.08)**5.265

    rho = pressure/(.2869 * (temp + 273.1)) * units.kg/(units.meter**3)

    return rho
