import pint

from RCPlanePerformance.IBattery import IBattery
from RCPlanePerformance.Units import units


class SimpleLiPoBattery(IBattery):
    def __init__(self, num_cells: int):
        self.num_cells = num_cells
        self.volts_per_cell = 3.7 * units.volt

    def get_voltage(self) -> pint.unit:
        return self.num_cells * self.volts_per_cell
