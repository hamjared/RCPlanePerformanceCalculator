from abc import ABC, abstractmethod

import pint


class IBattery(ABC):
    @abstractmethod
    def get_voltage(self) -> pint.unit:
        pass


