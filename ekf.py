import numpy as np
from quaternion import Quaternion

def Omega(omega):
    "Кососимметрическая матрица для кинематики кватерниона"
    wx, wy, wz, = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])


def skew(v):
    "Кососимметрическая матрица 3x3"
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

class EKF:
    def __init__(self, initial_quat, sigma_g, sigma_b, mag_sigma, sun_sigma, dt):
        self.q_est = initial_quat.normalize()   # оценка ориентации
        self.b_est = np.zeros(3)                # оценка дрейфа гироскопа
        self.P = np.diag([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])                # начальная ковариация ошибки

        # Параметры шумов
        self.sigma_g = 0.015
        self.sigma_b = 0.0002
        self.dt = dt

        # Матрицы Q и R 
        self.Q = self._compute_Q()
        self.R_mag = (mag_sigma**2) * np.eye(3)
        self.R_sun = (sun_sigma**2) * np.eye(3)
        self.R = np.block([
            [self.R_mag, np.zeros((3,3))],
            [np.zeros((3,3)), self.R_sun]
        ])

        # История для анализа
        self.history_innov_mag = []   # невязки магнитометра
        self.history_innov_sun = []   # невязки солнечного датчикa
        self.history_P_diag = []      # диагональные элементы P
        self.history_b_est = []       # оценка дрейфа
        self.history_q_est = []       # оценка кватерниона

    def _compute_Q(self):
        "Диагональное приближение для Q"
        dt = self.dt
        sigma_g2 = self.sigma_g**2
        sigma_b2 = self.sigma_b**2
        Q_theta = (sigma_g2 * dt + sigma_b2 * dt**3 / 3) * np.eye(3)
        Q_b = sigma_b2 * dt * np.eye(3)
        Q = np.block([
            [Q_theta, np.zeros((3,3))],
            [np.zeros((3,3)), Q_b]
        ])
        return Q

    def predict(self, gyro_meas):
        # 1. Компенсация дрейфа
        omega_hat = gyro_meas - self.b_est

        # 2. Интегрирование кватерниона (RK4)
        self.q_est = self._integrate_quat_rk4(self.q_est, omega_hat, self.dt)

        # 3. Матрица перехода ошибки F
        F = np.eye(6)
        F[:3, :3] = np.eye(3)  
        F[:3, 3:] = -np.eye(3) * self.dt

        # 4. Обновление ковариации
        self.P = F @ self.P @ F.T + self.Q

        P_epsilon = np.diag([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12])
        self.P = self.P + P_epsilon        
        
        self.history_b_est.append(self.b_est.copy())
        self.history_P_diag.append(np.diag(self.P).copy())
        self.history_q_est.append(self.q_est)

    def update(self, mag_meas=None, sun_meas=None):
        # Собираем измерения и опорные вектора
        z_list = []
        z_pred_list = []
        H_list = []

        if mag_meas is not None:
            z_pred_mag = self.q_est.rotate_vector_inverse(self.mag_ref) 
            z_list.append(mag_meas)
            z_pred_list.append(z_pred_mag)
            H_mag = np.hstack([skew(z_pred_mag), np.zeros((3,3))])
            H_list.append(H_mag)

        if sun_meas is not None:
            z_pred_sun = self.q_est.rotate_vector_inverse(self.sun_ref)
            z_list.append(sun_meas)
            z_pred_list.append(z_pred_sun)
            H_sun = np.hstack([skew(z_pred_sun), np.zeros((3,3))])
            H_list.append(H_sun)

        if not z_list:
            return  # нет измерений для обновления

        # Формируем общий вектор невязки, H и R
        z = np.concatenate(z_list)
        z_pred = np.concatenate(z_pred_list)
        H = np.vstack(H_list)

        # Выбираем нужные строки из R
        R = self.R[:len(z), :len(z)]  
        
        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        if mag_meas is not None:
            y_mag = mag_meas - z_pred_mag
            self.history_innov_mag.append(y_mag)

        if sun_meas is not None:
            y_sun = sun_meas - z_pred_sun
            self.history_innov_sun.append(y_sun)

        # Извлекаем поправки
        dtheta = dx[:3]
        dbias = dx[3:6]

        # Обновляем состояние
        norm_dtheta = np.linalg.norm(dtheta)
        if norm_dtheta > 1e-12:
            axis = dtheta / norm_dtheta
            delta_q = Quaternion.from_axis_angle(axis, norm_dtheta)
        else:
            delta_q = Quaternion(1, 0, 0, 0)
        self.q_est = (self.q_est * delta_q).normalize()
        self.b_est += dbias

        # Обновляем ковариацию (форма Джозефа)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        
    def _integrate_quat_rk4(self, q, omega, dt):
        # Преобразуем Quaternion в массив [w, x, y, z]
        q_arr = np.array([q.w, q.x, q.y, q.z], dtype=float)
        
        def dqdt(qq_arr):
            return 0.5 * Omega(omega) @ qq_arr
        
        k1 = dqdt(q_arr)
        k2 = dqdt(q_arr + 0.5 * dt * k1)
        k3 = dqdt(q_arr + 0.5 * dt * k2)
        k4 = dqdt(q_arr + dt * k3)
        
        q_new_arr = q_arr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Нормализация для предотвращения дрейфа нормы
        q_new_arr = q_new_arr / np.linalg.norm(q_new_arr)
        
        return Quaternion(q_new_arr[0], q_new_arr[1], q_new_arr[2], q_new_arr[3])
