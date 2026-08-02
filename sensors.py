import math
import numpy as np
import os
print("Текущая директория:", os.getcwd())
class Gyroscope:
    def __init__(self, sigma_g=0.01, sigma_b = 0.0001):
        self.sigma_g = sigma_g   # рад/с
        self.sigma_b = sigma_b   # рад/с/√с (Rate Random Walk)
    
    def measure(self, true_omega, bias, dt):
        noise = np.random.normal(0, self.sigma_g, 3)
        return true_omega + bias + noise
    
    def update_bias(self, bias, dt):
        bias_noise = np.random.normal(0, self.sigma_b * np.sqrt(dt), 3)
        return bias + bias_noise

class Magnetometer:
    def __init__(self, ref_vector, sigma_m):
        self.ref = np.array(ref_vector)   # опорный вектор в инерциальной системе
        self.sigma = sigma_m              # стандартное отклонение шума

    def maesure(self, true_quaternoin):
        v_body = true_quaternoin.rotate_vector_inverse(self.ref)
        noise = np.random.normal(0, self.sigma, 3)
        return v_body + noise

class SunSensor:
    def __init__(self, ref_vector, sigma_s):
        self.ref = np.array(ref_vector)   # единичный опорный вектор
        self.sigma = sigma_s              # стандартное отклонение шума (в компонентах)
       
    def measure(self, true_quaternion):
        v_body = true_quaternion.rotate_vector_inverse(self.ref)
        noise = np.random.normal(0, self.sigma,3)
        v = v_body + noise
        return v / np.linalg.norm(v)

