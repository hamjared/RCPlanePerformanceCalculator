import math
import string

import pint
from aerosandbox import Wing as AeroPyWing, WingXSec, Airfoil
from vector3d.point import Point

from RCPlanePerformance.Units import units
from RCPlanePerformance.IWing import IWing


class AeroPyTaperedIWing(IWing):
    def __init__(self, wing_span: pint.unit,
                 root_le_location: Point,
                 tip_chord: pint.unit,
                 root_chord: pint.unit,
                 wing_sweep: pint.unit,
                 name: string = 'wing',
                 airfoil: Airfoil = Airfoil(name="naca2412"),
                 is_vertical_stabilizer: bool = False):
        super().__init__(wing_span, root_le_location)
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.wing_sweep = wing_sweep
        self.name = name
        self.airfoil = airfoil
        self.is_vertical_stabilizer = is_vertical_stabilizer
        self.aeropy_wing = None

    def mean_aerodynamic_chord(self) -> pint.unit:
        t = self.tip_chord / self.root_chord  # taper ratio
        mac = self.root_chord * 2 / 3 * (1 + t + t ** 2) / (1 + t)
        return mac

    def get_wing_area(self) -> pint.unit:
        return (self.tip_chord + self.root_chord) / 2 * self.wing_span

    def get_tip_relative_to_root(self) -> Point:
        if self.is_vertical_stabilizer:
            z_le = self.wing_span
            y_le = 0 * units.inch
            x_le = z_le * math.tan(self.wing_sweep.to('radian').magnitude)

        else:
            y_le = self.wing_span / 2
            x_le = y_le * math.tan(self.wing_sweep.to('radian').magnitude)
            z_le = 0 * units.inch  # no dihedral/ z is 0 relative to wing datum

        return Point(x_le, y_le, z_le)

    def get_aero_py_wing(self) -> AeroPyWing:
        wing = AeroPyWing(
            name=self.name,
            x_le=-self.root_le_location.x.to('m').magnitude,  # Coordinates of the wing's leading edge
            y_le=-self.root_le_location.y.to('m').magnitude,
            z_le=self.root_le_location.z.to('m').magnitude,
            symmetric=(not self.is_vertical_stabilizer),
            xsecs=[  # The wing's cross ("X") sections
                WingXSec(  # Root
                    x_le=0,
                    y_le=0,
                    z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=self.root_chord.to('m').magnitude,
                    twist=0,  # degrees
                    airfoil=self.airfoil,
                    control_surface_type='asymmetric',
                    # Flap # Control surfaces are applied between a given XSec and the next one.
                    control_surface_deflection=0,  # degrees
                    control_surface_hinge_point=0.75  # as chord fraction
                ),
                WingXSec(  # Tip
                    x_le=-self.get_tip_relative_to_root().x.to('m').magnitude,
                    y_le=-self.get_tip_relative_to_root().y.to('m').magnitude,
                    z_le=self.get_tip_relative_to_root().z.to('m').magnitude,
                    chord=self.tip_chord.to('m').magnitude,
                    twist=0,
                    airfoil=self.airfoil
                )

            ]
        )
        self.aeropy_wing = wing
        return wing
