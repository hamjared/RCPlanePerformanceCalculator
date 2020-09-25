import pint


class LandingGear:
    def __init__(self, diameter: pint.unit, width: pint.unit, reference_area: pint.unit):
        self.__diameter = diameter
        self.__width = width
        self.__reference_area = reference_area
        self.recalc = True
        self.drag_coefficient = self.calc_drag_coefficient()


    @property
    def diameter(self):
        return self.__diameter

    @diameter.setter
    def diameter(self, diameter: pint.unit):
        self.__diameter = diameter
        self.recalc = True

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width: pint.unit):
        self.__width = width
        self.recalc = True

    @property
    def reference_area(self):
        return self.__reference_area

    @reference_area.setter
    def reference_area(self, reference_area: pint.unit):
        self.__reference_area = reference_area
        self.recalc = True

    def calc_drag_coefficient(self):
        if not self.recalc:
            return self.drag_coefficient
        self.recalc = False
        return 1.01 * self.__diameter * self.__width / self.reference_area
