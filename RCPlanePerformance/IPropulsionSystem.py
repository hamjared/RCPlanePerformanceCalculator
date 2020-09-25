from abc import ABC, abstractmethod

import pint


class IPropulsionSystem(ABC):
    @abstractmethod
    def get_thrust(self, velocity: pint.unit, altitude: pint.unit, throttlePercent: float = 1) -> pint.unit:
        pass

    @abstractmethod
    def get_power_required(self, velocity: float, altiude: float, throttlePercent: float):
        pass