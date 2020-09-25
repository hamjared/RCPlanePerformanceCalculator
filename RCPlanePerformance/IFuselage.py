from abc import ABC, abstractmethod


class IFuselage(ABC):
    @abstractmethod
    def calc_drag_coefficient(self):
        pass
