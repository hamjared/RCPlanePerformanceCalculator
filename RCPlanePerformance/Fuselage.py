import pint

from RCPlanePerformance.IFuselage import IFuselage


class Fuselage(IFuselage):

    def __init__(self, length: pint.unit, diameter: pint.unit, skin_friction_coefficient: float, wetted_area: pint.unit,
                 reference_area: pint.unit):
        self.length = length
        self.diameter = diameter
        self.skin_friction_coefficient = skin_friction_coefficient
        self.wetted_area = wetted_area
        self.reference_area = reference_area


    def calc_drag_coefficient(self):
        # https://mypages.iit.edu/~vural/RC%20Airplane%20Design.pdf
        fineness_ratio = self.length / self.diameter
        ff = 1 + 60 / (fineness_ratio ** 3) + 0.0025 * fineness_ratio
        return ff*self.skin_friction_coefficient*self.wetted_area/self.reference_area

