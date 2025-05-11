import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Parámetros del circuito
R = 1000      # ohmios
C = 0.001     # faradios
V_fuente = 5  # voltios

# EDO: dV/dt = (1/RC) * (V_fuente - V)
def f(t, V):
    return (1 / (R * C)) * (V_fuente - V)

# Condiciones iniciales
t0 = 0
V0 = 0
tf = 5
n = 20
h = (tf - t0) / n

# Inicialización de listas
t_vals = [t0]
V_aprox = [V0]
V_analitica = [V_fuente * (1 - np.exp(-t0 / (R * C)))]

# Método de Euler
t = t0
V = V0
for i in range(n):
    V = V + h * f(t, V)
    t = t + h
    t_vals.append(t)
    V_aprox.append(V)
    V_analitica.append(V_fuente * (1 - np.exp(-t / (R * C))))

# Guardar resultados
df = pd.DataFrame({
    "t": t_vals,
    "V_aproximada": V_aprox,
    "V_analitica": V_analitica
})
df.to_csv("resultado_RC_Euler.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, V_aprox, 'o-', label='Euler (Aproximada)', color='blue')
plt.plot(t_vals, V_analitica, 'x--', label='Analítica', color='red')
plt.title('Carga de un Capacitor - Método de Euler')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.savefig("comparacion_RC.png")
plt.show()
