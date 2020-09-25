from abc import ABC, abstractmethod
from aerosandbox import Wing as AeroPyWing

import pint
from vector3d.point import Point


class IWing(ABC):

    def __init__(self, wing_span: pint.unit, root_le_location: Point):
        self.wing_span = wing_span
        self.root_le_location = root_le_location

    def aspect_ratio(self) -> pint.unit:
        return self.wing_span ** 2 / self.get_wing_area()

    @abstractmethod
    def mean_aerodynamic_chord(self) -> pint.unit:
        pass

    @abstractmethod
    def get_wing_area(self) -> pint.unit:
        pass

    @abstractmethod
    def get_aero_py_wing(self) -> AeroPyWing:
        pass
