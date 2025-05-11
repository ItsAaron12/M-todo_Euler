import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
T0 = 90           # Temperatura inicial (°C)
T_amb = 25        # Temperatura ambiente (°C)
k = 0.07          # Constante de enfriamiento
t0 = 0            # Tiempo inicial (min)
tf = 30           # Tiempo final (min)
n = 30            # Número de pasos
h = (tf - t0) / n # Tamaño del paso

# EDO: dT/dt = -k(T - T_amb)
def f(t, T):
    return -k * (T - T_amb)

# Inicialización de listas
t_vals = [t0]
T_aprox = [T0]
T_analitica = [T_amb + (T0 - T_amb) * np.exp(-k * t0)]

# Método de Euler
t = t0
T = T0
for i in range(n):
    T = T + h * f(t, T)
    t = t + h
    t_vals.append(t)
    T_aprox.append(T)
    T_analitica.append(T_amb + (T0 - T_amb) * np.exp(-k * t))

# Guardar resultados
df = pd.DataFrame({
    "t": t_vals,
    "T_aproximada": T_aprox,
    "T_analitica": T_analitica
})
df.to_csv("enfriamiento_euler.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, T_aprox, 'o-', label='Euler (Aproximada)', color='blue')
plt.plot(t_vals, T_analitica, 'x--', label='Analítica', color='red')
plt.title('Enfriamiento de un cuerpo - Ley de Newton')
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.legend()
plt.savefig("enfriamiento_comparacion.png")
plt.show()
