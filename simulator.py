import numpy as np
from quaternion import Quaternion
from sensors import Gyroscope, Magnetometer, SunSensor

def Omega(omega):
    "Кососимметрическая матрица для кинематики кватерниона"
    wx, wy, wz, = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

class SatelliteSimulator:
    def __init__(self, initial_quat, true_omega,
                 gyro_sigma_g=0.01, gyro_sigma_b=0.0001,
                 mag_ref=(1,0,0), mag_sigma=0.05,
                 sun_ref=(0,0,1), sun_sigma=0.02,
                 dt=0.01):
        self.dt = dt
        self.true_q = initial_quat.normalize()
        self.true_omega = np.array(true_omega)
        self.true_bias = np.zeros(3)

        self.gyro = Gyroscope(gyro_sigma_g, gyro_sigma_b)
        self.mag = Magnetometer(mag_ref, mag_sigma)
        self.sun = SunSensor(sun_ref, sun_sigma)

        self.history ={
            'time': [],
            'true_q': [],
            'gyro': [],
            'mag': [],
            'sun': [],
            'true_bias': []
        }
        self.step_counter = 0

    def step(self):
        dt = self.dt
        self.step_counter +=1
        current_time = self.step_counter * dt

        # 1. Обновление истинного дрейфа (случайное блуждание)
        self.true_bias = self.gyro.update_bias(self.true_bias, dt)

        # 2. Генерация измерений
        gyro_meas = self.gyro.measure(self.true_omega, self.true_bias, dt)
        mag_meas = self.mag.maesure(self.true_q)
        sun_meas = self.sun.measure(self.true_q)

        # 3. Интегрирование истинного кватерниона (RK4)
        self.true_q = self._integrate_quat_rk4(self.true_q, self.true_omega, dt)

        # 4. Сохранение истории
        self.history['time'].append(current_time)
        self.history['true_q'].append(self.true_q)
        self.history['gyro'].append(gyro_meas)
        self.history['mag'].append(mag_meas)
        self.history['sun'].append(sun_meas)
        self.history['true_bias'].append(self.true_bias.copy())

        # Возвращаем измерения для подачи в EKF
        return {
            'gyro': gyro_meas,
            'mag': mag_meas,
            'sun': sun_meas
        }

    def _integrate_quat_rk4(self, q, omega, dt):
        # Преобразуем кватернион в массив
        q_arr = np.array([q.w, q.x, q.y, q.z])
        
        def dqdt(qq_arr):
            return 0.5 * Omega(omega) @ qq_arr
        
        k1 = dqdt(q_arr)
        k2 = dqdt(q_arr + 0.5 * dt * k1)
        k3 = dqdt(q_arr + 0.5 * dt * k2)
        k4 = dqdt(q_arr + dt * k3)
        
        q_new_arr = q_arr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Нормализуем массив 
        q_new_arr = q_new_arr / np.linalg.norm(q_new_arr)
        
        return Quaternion(q_new_arr[0], q_new_arr[1], q_new_arr[2], q_new_arr[3])


