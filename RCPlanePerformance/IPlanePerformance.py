from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pint


class IPlanePerformance(ABC):
    @abstractmethod
    def get_top_speed(self):
        pass

    @abstractmethod
    def get_stall_speed(self):
        pass

    @abstractmethod
    def get_flight_time(self):
        pass

    @abstractmethod
    def get_take_off_roll_distance(self):
        pass

    @abstractmethod
    def get_thrust_required_curve(self, altitude: pint.unit, min_velocity: pint.unit, max_velocity: pint.unit,
                                  num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        pass
