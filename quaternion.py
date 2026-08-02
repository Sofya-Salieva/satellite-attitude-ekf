import math
import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def rotate_vector_inverse(self, v):
        """Поворот вектора из инерциальной системы в связанную (R^T @ v)"""
        return self.conjugate().rotate_vector_matrix(v)

    def __mul__(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def norm(self):
        return math.sqrt(self.w**2 + self.x**2+self.y**2+self.z**2)
    
    def normalize(self):
        n =  self.norm()
        if n < 1e-12:
            return Quaternion(1, 0, 0, 0)   # безопасное значение «нет вращения»
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)
    
    def conjugate(self):
       return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self):
        n2 = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n2 < 1e-12:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w / n2, -self.x / n2, -self.y / n2, -self.z / n2)

    @staticmethod
    def from_axis_angle(axis, angle):
        # axis - единичный вектор
        half = angle / 2.0
        sin_half = math.sin(half)
        return Quaternion(
            math.cos(half),
            axis[0]*sin_half,
            axis[1]*sin_half,
            axis[2]*sin_half
        )
    
    def to_rotation_matrix(self):
        n = self.normalize()
        w, x, y, z = n.w, n.x, n.y, n.z
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
        ])
        return R
    
    def rotate_vector_matrix(self, v):
        R = self.to_rotation_matrix()
        return R @ np.array(v)
    
    def rotate_vector_effect(self, v):
        q = self.normalize()
        # v -- трёхмерный вектор, превращаем в чисто векторный кватернион
        v_quat = Quaternion(0, v[0], v[1], v[2])
        result = self * v_quat * self.inverse()
        return np.array([result.x, result.y, result.z])

