import numpy as np
import matplotlib.pyplot as plt
from simulator import SatelliteSimulator
from ekf import EKF
from quaternion import Quaternion

# Параметры симуляции 
dt = 0.01                      # шаг по времени (сек)
T_total = 120                 # общее время симуляции (сек)
steps = int(T_total / dt)

# Истинная угловая скорость (рад/с)
true_omega = np.array([0.1, 0.02, -0.05])

# Начальный истинный кватернион (единичный)
q0_true = Quaternion(1, 0, 0, 0)

# Параметры шумов датчиков
sigma_g = 0.01
sigma_b = 0.0001
sigma_mag = 0.05
sigma_sun = 0.02

def Omega(omega):
    "Кососимметрическая матрица для кинематики кватерниона"
    wx, wy, wz, = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

# Создание симулятора
sim = SatelliteSimulator(
    initial_quat=q0_true,
    true_omega=true_omega,
    gyro_sigma_g=sigma_g,
    gyro_sigma_b=sigma_b,
    mag_ref=(1, 0, 0),
    mag_sigma=sigma_mag,
    sun_ref=(0, 0, 1),
    sun_sigma=sigma_sun,
    dt=dt
)

# Создание EKF
q0_est = Quaternion(1, 0, 0, 0)

ekf = EKF(
    initial_quat=q0_est,
    sigma_g=sigma_g,
    sigma_b=sigma_b,
    mag_sigma=sigma_mag,
    sun_sigma=sigma_sun,
    dt=dt
)

ekf.mag_ref = np.array([1, 0, 0])
ekf.sun_ref = np.array([0, 0, 1])

# История для анализа
q_dr = q0_est
b_dr = np.zeros(3)

errors_ekf = []
errors_dr = []
time_hist = []

history_est_q = []    # оценки кватерниона
history_true_q = []   # истинные кватернионы из симулятора
history_time = []

# Основной цикл
for step_idx in range(steps):
    # Получить измерения от симулятора
    meas = sim.step()       

    # Измерения гироскопа поступают каждый шаг 
    ekf.predict(meas['gyro'])

    # Магнитометр и солнечный датчик обновляются реже 
    if step_idx % 10 == 0:
        ekf.update(mag_meas=meas['mag'], sun_meas=meas['sun'])

    # Сохраняем состояние для последующего анализа
    history_est_q.append(ekf.q_est)
    history_true_q.append(sim.true_q)   
    history_time.append(sim.step_counter * dt)

    gyro_raw = meas['gyro']
    def dr_dqdt(qq_arr):
        return 0.5 * Omega(gyro_raw) @ qq_arr

    q_arr = np.array([q_dr.w, q_dr.x, q_dr.y, q_dr.z])
    k1 = dr_dqdt(q_arr)
    k2 = dr_dqdt(q_arr + 0.5*dt*k1)
    k3 = dr_dqdt(q_arr + 0.5*dt*k2)
    k4 = dr_dqdt(q_arr + dt*k3)
    q_arr_new = q_arr + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    q_arr_new = q_arr_new /np.linalg.norm(q_arr_new)
    q_dr = Quaternion(q_arr_new[0], q_arr_new[1], q_arr_new[2], q_arr_new[3])

    q_true = sim.true_q

    q_err_ekf = q_true * ekf.q_est.inverse()
    angle_ekf = 2* np.arccos(min(abs(q_err_ekf.w), 1.0)) * 180/np.pi
    errors_ekf.append(angle_ekf)

    q_err_dr = q_true * q_dr.inverse()
    angle_dr = 2* np.arccos(min(abs(q_err_dr.w), 1.0)) * 180/np.pi
    errors_dr.append(angle_dr)
    time_hist.append(step_idx * dt)

# Графики ошибок
fig, axes = plt.subplots(2,3, figsize=(15,10))
ax = axes[0,0]
plt.plot(history_time, errors_ekf)
plt.xlabel('Время (с)')
plt.ylabel('Ошибка ориентации (град)')
plt.title('Угловая ошибка EKF')
plt.grid(True)

# 1. Ошибка ориентации EKF vs Dead Reckoning
ax = axes[0, 0]
ax.plot(time_hist, errors_ekf, label='EKF')
ax.plot(time_hist, errors_dr, label='Только гироскоп (Dead Reckoning)')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Ошибка (градусы)')
ax.set_title('Ошибка ориентации')
ax.legend()
ax.grid()

# 2. Невязки магнитометра
ax = axes[0, 1]
innov_mag = np.array(ekf.history_innov_mag)  
times_update = np.arange(0, len(innov_mag)) * dt * 10  
for i in range(3):
    ax.plot(times_update, innov_mag[:, i], label=f'ось {i}')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Невязка')
ax.set_title('Невязки магнитометра')
ax.legend()
ax.grid()

# 3. Невязки солнечного датчика
ax = axes[0, 2]
innov_sun = np.array(ekf.history_innov_sun)
for i in range(3):
    ax.plot(times_update, innov_sun[:, i], label=f'ось {i}')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Невязка')
ax.set_title('Невязки солнечного датчика')
ax.legend()
ax.grid()

# 4. Оценка дрейфа гироскопа
ax = axes[1, 0]
true_bias = np.array(sim.history['true_bias'])
times_all = np.array(sim.history['time'])
ax.plot(times_all, true_bias[:, 0], 'b-', label='истинный bias X')
ax.plot(times_all, true_bias[:, 1], 'g-', label='истинный bias Y')
ax.plot(times_all, true_bias[:, 2], 'r-', label='истинный bias Z')

b_est_arr = np.array(ekf.history_b_est)
ax.plot(times_all, b_est_arr[:, 0], 'b--', label='оценка bias X')
ax.plot(times_all, b_est_arr[:, 1], 'g--', label='оценка bias Y')
ax.plot(times_all, b_est_arr[:, 2], 'r--', label='оценка bias Z')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Дрейф (рад/с)')
ax.set_title('Оценка дрейфа гироскопа')
ax.legend()
ax.grid()

# 5. Диагональные элементы P
ax = axes[1, 1]
P_diag = np.array(ekf.history_P_diag)  
for i in range(6):
    ax.plot(times_all, P_diag[:, i], label=f'P[{i},{i}]')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Значение')
ax.set_title('Диагональные элементы ковариационной матрицы P')
ax.legend()
ax.grid()

# 6. Границы 3σ для угловой ошибки 
ax = axes[1, 2]
sigma_theta = np.sqrt(P_diag[:, 0])  

err_upd = [errors_ekf[i] for i in range(len(errors_ekf)) if i % 10 == 0][:len(sigma_theta)]
ax.plot(times_update, err_upd, label='Ошибка ориентации')
ax.plot(times_all, 3*sigma_theta*180/np.pi, 'r--', label='3σ граница')
ax.plot(times_all, -3*sigma_theta*180/np.pi, 'r--')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Ошибка (градусы)')
ax.set_title('Ошибка и границы 3σ')
ax.legend()
ax.grid()

plt.tight_layout()
plt.show()
