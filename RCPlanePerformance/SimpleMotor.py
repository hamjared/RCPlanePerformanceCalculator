import pint

from RCPlanePerformance.IMotor import IMotor


class SimpleMotor(IMotor):
    def __init__(self, kv: pint.unit):
        self.kv = kv

    def get_rpm(self, voltage: pint.unit) -> pint.unit:
        return self.kv * voltage
