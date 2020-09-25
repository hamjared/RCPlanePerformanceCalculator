from abc import ABC, abstractmethod

import pint


class IPropeller(ABC):
    @abstractmethod
    def calculate_thrust(self, angular_velocity: pint.unit, aircraft_velocity: pint.unit) -> pint.unit:
        pass

    @abstractmethod
    def calculate_power(self, angular_velocity: pint.unit, aircraft_velocity: pint.unit) -> pint.unit:
        pass