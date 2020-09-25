import pint

from RCPlanePerformance import IBattery, IMotor, IPropeller
from RCPlanePerformance.IPropulsionSystem import IPropulsionSystem
from typing import Type


class ElectricPropulsionSystem(IPropulsionSystem):

    def __init__(self, battery , motor, propeller):
        self.battery = battery
        self.motor = motor
        self.propeller = propeller

    def get_thrust(self, velocity: pint.unit, altitude: pint.unit, throttlePercent: float = 1) -> pint.unit:
        # Equation taken from:
        # https://www.electricrcaircraftguy.com/2014/04/propeller-static-dynamic-thrust-equation-background.html
        rpm = self.motor.get_rpm(self.battery.get_voltage()) * throttlePercent
        return self.propeller.calculate_thrust(rpm, velocity, altitude)

    def get_power_required(self, velocity: float, altitude: float, throttlePercent: float) -> pint.unit:
        raise NotImplementedError("get power required has not been implemented")
