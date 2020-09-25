import string
from typing import Type, List, Tuple

import numpy as np
import pint
from aerosandbox import Airplane as AeroPyPlane, Airplane, OperatingPoint, Buildup
from vector3d.point import Point

from RCPlanePerformance import AtmosphericModel
from RCPlanePerformance.Fuselage import Fuselage
from RCPlanePerformance.IPlanePerformance import IPlanePerformance
from RCPlanePerformance.IPlanePolar import IPlanePolar
from RCPlanePerformance.IPropulsionSystem import IPropulsionSystem
from RCPlanePerformance.IWing import IWing
from RCPlanePerformance.LandingGear import LandingGear
from RCPlanePerformance.Units import units


class Plane(IPlanePerformance, IPlanePolar):

    def __init__(self, weight: pint.unit, main_wing: Type[IWing], horizontal_tail: Type[IWing],
                 vertical_tail: Type[IWing],
                 fuselage: Fuselage,
                 landing_gear: List[LandingGear],
                 propulsion_system: IPropulsionSystem,
                 cog: Point,
                 name: string = 'plane'):

        self.weight = weight
        self.main_wing = main_wing
        self.horizontal_tail = horizontal_tail
        self.vertical_tail = vertical_tail
        self.fuselage = fuselage
        self.landing_gear = landing_gear
        self.cog = cog
        self.name = name
        self.aeropy_model = None
        self.get_aeropy_model()
        self.propulsion_system = propulsion_system

    def get_polar(self, altitude: pint.unit, viscosity: pint.unit, velocity: pint.unit, alpha_start: pint.unit,
                  alpha_end: pint.unit, num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alphas = []
        CL = []
        CD = []
        CM = []
        for alpha in np.linspace(alpha_start.to('degree').magnitude, alpha_end.to('degree').magnitude, num=num_steps):
            ap = Buildup(  # Set up the AeroProblem
                airplane=self.get_aeropy_model(),
                op_point=OperatingPoint(
                    density=AtmosphericModel.get_air_density(altitude).to('kg/m**3').magnitude,  # kg/m^3
                    viscosity=1.81e-5,  # kg/m-s
                    velocity=velocity.to('m/s').magnitude,  # m/s
                    mach=0,  # Freestream mach number
                    alpha=alpha,  # In degrees
                    beta=0,  # In degrees
                    p=0,  # About the body x-axis, in rad/sec
                    q=0,  # About the body y-axis, in rad/sec
                    r=0,  # About the body z-axis, in rad/sec
                ),
            )
            CD_total = ap.CD + self.fuselage.calc_drag_coefficient()
            for lg in self.landing_gear:
                CD_total += lg.calc_drag_coefficient()

            alphas.append(alpha)
            CL.append(ap.CL)
            CD.append(CD_total)
            CM.append(ap.Cm)

        return np.array(alphas), np.array(CL), np.array(CD), np.array(CM)

    def get_aeropy_model(self) -> AeroPyPlane:
        if self.aeropy_model:
            return self.aeropy_model
        airplane = Airplane(
            x_ref=self.cog.x.to('m').magnitude,
            y_ref=self.cog.y.to('m').magnitude,
            z_ref=self.cog.z.to('m').magnitude,
            wings=[self.main_wing.get_aero_py_wing(), self.horizontal_tail.get_aero_py_wing(),
                   self.vertical_tail.get_aero_py_wing()]
        )
        return airplane

    def get_thrust_required_curve(self, altitude: pint.unit, min_velocity: pint.unit, max_velocity: pint.unit,
                                  num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        alpha, CL, CD, CM = self.get_polar(altitude=altitude,
                                           viscosity=1.81e-5 * units.kg / units.m / units.s,
                                           velocity=(min_velocity + max_velocity) / 2,
                                           alpha_start=-5 * units.degree,
                                           alpha_end=12 * units.degree,
                                           num_steps=100)
        rho = AtmosphericModel.get_air_density(altitude).to('kg/m**3').magnitude
        thrust_required = []
        velocities = []
        for velocity in np.linspace(min_velocity, max_velocity, num=num_steps):
            W = self.weight.to('N').magnitude
            v = velocity.to('m/s').magnitude
            S = self.main_wing.get_wing_area().to('m**2').magnitude
            CL_required = 2 * W / (rho * v ** 2 * S)
            CD_required = np.interp(CL_required, CL, CD)
            thrust_required.append(W / (CL_required / CD_required))
            velocities.append(v)

        return np.array(velocities) * units.meter / units.s, np.array(thrust_required) * units.N

    def get_power_required_curve(self,altitude: pint.unit, min_velocity: pint.unit, max_velocity: pint.unit,
                                  num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        v, t_r = self.get_thrust_required_curve(altitude, min_velocity, max_velocity, num_steps)
        return v, v * t_r

    def get_thrust_available_curve(self, altitude: pint.unit, min_velocity: pint.unit, max_velocity: pint.unit,
                                   num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        thrust_available = []
        velocities = []
        for velocity in np.linspace(min_velocity, max_velocity, num=num_steps):
            thrust_available.append(self.propulsion_system.get_thrust(velocity, altitude, 1).to('N').magnitude)
            velocities.append(velocity.to('m/s').magnitude)
        return np.array(velocities) * units.m/units.s, np.array(thrust_available)* units.N

    def get_top_speed(self):
        pass

    def get_stall_speed(self):
        pass

    def get_flight_time(self):
        pass

    def get_take_off_roll_distance(self):
        pass
