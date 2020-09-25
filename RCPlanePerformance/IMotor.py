from abc import ABC, abstractmethod

import pint


class IMotor(ABC):
    @abstractmethod
    def get_rpm(self, voltage: pint.unit) -> pint.unit:
        pass