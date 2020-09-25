import matplotlib.pyplot as plt
import numpy as np
from aerosandbox import Airfoil
from aerosandbox.library.airfoils import flat_plate
from vector3d.point import Point

from RCPlanePerformance.AeroPyTaperedWing import AeroPyTaperedIWing
from RCPlanePerformance.ElectricPropulsionSystem import ElectricPropulsionSystem
from RCPlanePerformance.Fuselage import Fuselage
from RCPlanePerformance.LandingGear import LandingGear
from RCPlanePerformance.Plane import Plane
from RCPlanePerformance.Propeller import Propeller
from RCPlanePerformance.SimpleBattery import SimpleLiPoBattery
from RCPlanePerformance.SimpleMotor import SimpleMotor
from RCPlanePerformance.Units import units

naca2412_self_defined = Airfoil(
    name="test",
    coordinates=None,
    CL_function=lambda alpha, Re, mach, deflection: (
            (alpha) * (0.09544) + 0.191
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (
            alpha ** 2 * 0.0002 - .0003 * alpha + 0.01
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (
        -0.05
    )
)

if __name__ == '__main__':
    naca2412 = Airfoil('naca2421')
    # naca2412.populate_sectional_functions_from_xfoil_fits()
    naca0009 = Airfoil('naca0009')
    # naca0009.populate_sectional_functions_from_xfoil_fits()
    main_wing = AeroPyTaperedIWing(
        wing_span=4 * units.feet,
        root_le_location=Point(0 * units.inch, 0 * units.inch, 0 * units.inch),
        tip_chord=7 * units.inch,
        root_chord=7 * units.inch,
        wing_sweep=0 * units.degree,
        name='main wing',
        airfoil=naca2412_self_defined,
        is_vertical_stabilizer=False

    )

    h_tail = AeroPyTaperedIWing(
        wing_span=1 * units.feet,
        root_le_location=Point(2 * units.feet, 0 * units.inch, 0 * units.inch),
        tip_chord=4 * units.inch,
        root_chord=4 * units.inch,
        wing_sweep=0 * units.degree,
        name='h tail',
        airfoil=flat_plate,
        is_vertical_stabilizer=False

    )

    v_tail = AeroPyTaperedIWing(
        wing_span=0.5 * units.feet,
        root_le_location=Point(2 * units.feet, 0 * units.inch, 0 * units.inch),
        tip_chord=4 * units.inch,
        root_chord=4 * units.inch,
        wing_sweep=30 * units.degree,
        name='v tail',
        airfoil=flat_plate,
        is_vertical_stabilizer=True

    )

    fuse = Fuselage(length=3 * units.feet,
                    diameter=10 * units.inch,
                    skin_friction_coefficient=0.0041,
                    wetted_area=10 * units.feet ** 2,
                    reference_area=main_wing.get_wing_area())

    lg = LandingGear(diameter=2.5 * units.inch,
                     width=0 * units.inch,
                     reference_area=main_wing.get_wing_area())

    propeller = Propeller(
        diameter=10 * units.inch,
        pitch=7 * units.inch,
        ct_function=lambda J: -.1118 * J ** 2 - 0.0255 * J + 0.114 if (J < 0.4) else -.201 * J + 0.1659

    )
    battery = SimpleLiPoBattery(3)
    motor = SimpleMotor(900 * units.rpm / units.volt)
    propulsionSystem = ElectricPropulsionSystem(battery, motor, propeller)
    plane = Plane(
        weight=4 * units.lbf,
        main_wing=main_wing,
        horizontal_tail=h_tail,
        vertical_tail=v_tail,
        fuselage=fuse,
        landing_gear=[lg, lg, lg],
        cog=Point(2.5 * units.inch, 0 * units.inch, 0 * units.inch),
        propulsion_system=propulsionSystem
    )

    # plane.get_aeropy_model().draw()

    alpha, CL, CD, CM = plane.get_polar(altitude=0 * units.feet,
                                        viscosity=1.81e-5 * units.kg / units.m / units.s,
                                        velocity=20 * units.mph,
                                        alpha_start=-5 * units.degree,
                                        alpha_end=12 * units.degree,
                                        num_steps=50

                                        )
    print("fuse CD: " + str(plane.fuselage.calc_drag_coefficient().to('dimensionless')))
    print("lg CD: " + str(3 * plane.landing_gear[0].calc_drag_coefficient()))
    fig, ax = plt.subplots()
    ax.plot(CL, CD)

    ax.set(xlabel='CL', ylabel='CD',
           title='C_L C_D')

    ax.grid()

    plt.show()

    print("Interpolated CD at CL of 0.2 : {}".format(np.interp(0.2, CL, CD)))
    v, t_r = plane.get_thrust_required_curve(altitude=0 * units.feet,
                                             min_velocity=15 * units.mph,
                                             max_velocity=70 * units.mph,
                                             num_steps=50)

    v_t_a, t_a = plane.get_thrust_available_curve(altitude=0 * units.feet,
                                                  min_velocity=15 * units.mph,
                                                  max_velocity=70 * units.mph,
                                                  num_steps=50)

    fig, ax = plt.subplots()
    ax.plot(v.to('mph'), t_r.to('lbf'))
    ax.plot(v_t_a.to('mph'), t_a.to('lbf'))

    ax.set(xlabel='velocity (mph)', ylabel='thrust (lbs)',
           title='Thrust Required')

    ax.grid()

    plt.show()

    v, p_r = plane.get_power_required_curve(altitude=0 * units.feet,
                                            min_velocity=15 * units.mph,
                                            max_velocity=70 * units.mph,
                                            num_steps=50)

    fig, ax = plt.subplots()
    power_required = t_r * v
    ax.plot(v.to('mph'), p_r.to('W'))

    ax.set(xlabel='velocity (mph)', ylabel='power (W)',
           title='Power Required')

    ax.grid()

    plt.show()
