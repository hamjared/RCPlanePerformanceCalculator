from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pint


class IPlanePolar(ABC):

    @abstractmethod
    def get_polar(self, altitude: pint.unit, viscosity: pint.unit, velocity: pint.unit, alpha_start: pint.unit,
                  alpha_end: pint.unit, num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

