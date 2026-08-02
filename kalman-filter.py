import math
import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

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

class Magnetomer:
    def __init__(self, ref_vector, sigma_m):
        self.ref = np.array(ref_vector)   # опорный вектор в инерциальной системе
        self.sigma = sigma_m              # стандартное отклонение шума

    def maesure(self, true_quaternoin):
        R=true_quaternoin.to_rotation_matrix()
        noise = np.random.normal(0, self.sigma, 3)
        return R @ self.ref + noise

class SunSensor:
    def __init__(self, ref_vector, sigma_s):
        self.ref = np.array(ref_vector)   # единичный опорный вектор
        self.sigma = sigma_s              # стандартное отклонение шума (в компонентах)
       
    def measure(self, true_quaternion):
        R = true_quaternion.to_rotation_matrix()
        noise = np.random.normal(0, self.sigma,3)
        v = R@self.ref + noise
        return v / np.linalg.norm(v)

q = Quaternion.from_axis_angle([0,0,1], math.radians(90))
identity = Quaternion(1,0,0,0)
result = q*q*q*q
print(result.w, result.x, result.y, result.z)

# Проверка 1: нулевой поворот
q_id = Quaternion(1, 0, 0, 0)
R_id =q_id.to_rotation_matrix()
print("Единичный кватернион:\n", R_id)
assert np.allclose(R_id, np.eye(3), atol=1e-10), "Ошибка: единичный кватернион не дал единичную матрицу"
print("Тест 1 пройден: нулевой поворот.\n")

# Проверка 2: поворот на 90° вокруг оси Z
q_z90 = Quaternion.from_axis_angle([0, 0, 1], math.radians(90))
R_z90 = q_z90.to_rotation_matrix()
expected_z90 = np.array([
    [0, -1,  0],
    [1,  0,  0],
    [0,  0,  1]
])
print("Поворот 90° вокруг Z:\n", R_z90)
assert np.allclose(R_z90, expected_z90, atol=1e-10), "Ошибка: поворот 90° вокруг Z"

# Дополнительно проверим на векторах
v = np.array([1., 0., 0.])
v_rot = R_z90 @ v
assert np.allclose(v_rot, [0., 1., 0.], atol=1e-10), "Ошибка: поворот вектора (1,0,0)"
print("Тест 2 пройден: поворот на 90° вокруг Z.\n")

# Проверка 3: поворот на 180° вокруг оси X
q_x180 = Quaternion.from_axis_angle([1, 0, 0], math.radians(180))
R_x180 = q_x180.to_rotation_matrix()
expected_x180 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
print("Поворот 180° вокруг X:\n", R_x180)
assert np.allclose(R_x180, expected_x180, atol=1e-10), "Ошибка: поворот 180° вокруг X"
print("Тест 3 пройден: поворот на 180° вокруг X.\n")

# Проверка 4: ортогональность и определитель
for angle in [30, 45, 120]:
    q = Quaternion.from_axis_angle([1, 0, 0], math.radians(angle))
    R = q.to_rotation_matrix()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), f"Ошибка ортогональности для угла {angle}"
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-10, f"Определитель {det} не равен 1 для угла {angle}"
print("Тест 4 пройден: матрицы ортогональны и det=1.\n")

print("Все тесты пройдены успешно!")

# Тест: оба метода дают одинаковый результат
v_test = np.array([1., 2., 3.])
q_test = Quaternion.from_axis_angle([0, 1, 0], math.radians(75))
rot1 = q_test.rotate_vector_matrix(v_test)
rot2 = q_test.rotate_vector_effect(v_test)
print("Поворот вектора матрицей:", rot1)
print("Поворот вектора матрицей:", rot2)

# Тест гироскопа: проверка шума
g = Gyroscope(sigma_g=0.01, sigma_b=0.0)
N = 100000
true_omega = np.zeros(3)
bias = np.zeros(3)
measurements = np.array([g.measure(true_omega, bias, dt=0.1) for _ in range(N)])
std_est = np.std(measurements, axis = 0)
print("Измеренное стандартное отклонение шума:", std_est)
print("Ожидаемое: 0.01")
assert np.allclose(std_est, 0.01, atol=0.001), "Ошибка: неверная амплитуда шума"
print("Тест шума гироскопа пройден.\n")

# Тест дрейфа
sigma_b = 0.001
dt = 0.1
steps = 10000
g_drift = Gyroscope(sigma_g=0, sigma_b=sigma_b)
b = np.zeros(3)
history = np.zeros((steps,3))
for i in range(steps):
    b = g_drift.update_bias(b, dt)
    history[i] = b

T = steps * dt
expected_std = sigma_b * np.sqrt(T)
actual_std = np.std(history[-1000:])
print(f"Стандартное отклонение дрейфа через {T} сек: {actual_std:.6f}, ожидаемое: {expected_std:.6f}")
import matplotlib.pyplot as plt
plt.plot(history)
plt.title("Дрейф гироскопа (компоненты)")
plt.show()