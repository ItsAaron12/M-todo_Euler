import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros físicos
g = 9.81          # gravedad (m/s^2)
m = 2             # masa (kg)
k = 0.5           # coef. de fricción lineal (kg/s)

# EDO: dv/dt = g - (k/m)v
def f(t, v):
    return g - (k / m) * v

# Condiciones iniciales
t0 = 0
v0 = 0
tf = 10
n = 50
h = (tf - t0) / n

# Inicialización de listas
t_vals = [t0]
v_aprox = [v0]
v_analitica = [ (m * g / k) * (1 - np.exp(- (k / m) * t0)) ]

# Método de Euler
t = t0
v = v0
for i in range(n):
    v = v + h * f(t, v)
    t = t + h
    t_vals.append(t)
    v_aprox.append(v)
    v_analitica.append((m * g / k) * (1 - np.exp(- (k / m) * t)))

# Guardar resultados en CSV
df = pd.DataFrame({
    "t": t_vals,
    "v_aproximada": v_aprox,
    "v_analitica": v_analitica
})
df.to_csv("caida_libre_resistencia.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, v_aprox, 'o-', label='Euler (Aproximada)', color='blue')
plt.plot(t_vals, v_analitica, 'x--', label='Analítica', color='red')
plt.title('Caída Libre con Resistencia del Aire')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True)
plt.legend()
plt.savefig("caida_libre_comparacion.png")
plt.show()
