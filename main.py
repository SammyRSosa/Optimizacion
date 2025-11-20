import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import time

# --- 1. Definición del Modelo ---
def f(x):
    # x es un vector [x1, x2]
    # Protegemos contra overflow en exp
    if x[0] > 100: return np.inf 
    return np.exp(x[0]) * np.arctan(x[0]**2 + x[1]**2)

def gradient(x):
    x1, x2 = x[0], x[1]
    u = x1**2 + x2**2
    denom = 1 + u**2
    # Derivadas parciales
    df_dx = np.exp(x1) * np.arctan(u) + np.exp(x1) * (2*x1 / denom)
    df_dy = np.exp(x1) * (2*x2 / denom)
    return np.array([df_dx, df_dy])

# --- 2. Algoritmos de Clase ---

# A) Máximo Descenso (Gradient Descent) con Armijo
def maximo_descenso(start_point, max_iter=2000, tol=1e-6):
    x = np.array(start_point, dtype=float)
    path = [x.copy()]
    start_time = time.time()
    
    for i in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol: break
        
        p = -grad # Dirección de máximo descenso
        
        # Armijo (Backtracking)
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        while f(x + alpha*p) > f(x) + c * alpha * np.dot(grad, p):
            alpha *= rho
            if alpha < 1e-10: break 
            
        x = x + alpha * p
        path.append(x.copy())
        
    return x, i, path, f(x), time.time() - start_time

# B) Método de Newton (Puro, enseñado en clase)
def newton_method(start_point, max_iter=1000, tol=1e-6):
    x = np.array(start_point, dtype=float)
    path = [x.copy()]
    start_time = time.time()
    n = len(x)
    
    for i in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol: break
        
        # Hessiano por diferencias finitas (para simular cálculo)
        H = np.zeros((n, n)); h_step = 1e-5
        for j in range(n):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[j] += h_step; x_minus[j] -= h_step
            H[:, j] = (gradient(x_plus) - gradient(x_minus)) / (2 * h_step)
            
        try:
            p = np.linalg.solve(H, -grad)
        except:
            p = -grad # Fallback si H es singular
            
        # Armijo
        alpha = 1.0; rho = 0.5; c = 1e-4
        while f(x + alpha*p) > f(x) + c * alpha * np.dot(grad, p):
            alpha *= rho
            if alpha < 1e-10: break
            
        x = x + alpha * p
        path.append(x.copy())
        
    return x, i, path, f(x), time.time() - start_time

# C) Quasi-Newton (BFGS) - Algoritmo de Librería (Permitido y Recomendado)
def quasi_newton_bfgs(start_point):
    path = [np.array(start_point)]
    start_time = time.time()
    
    def callback(xk):
        path.append(xk)
        
    res = minimize(f, start_point, method='BFGS', jac=gradient, callback=callback, tol=1e-6)
    return res.x, res.nit, path, res.fun, time.time() - start_time

# --- 3. Experimentación Masiva (Puntos -100 a 100) ---
print("Ejecutando experimentos...")
results = []
# Reducimos un poco la malla para que corra rápido ahora, pero cubre el rango
grid = range(-10, 11, 2) # x10 son -100 a 100

for i in grid:
    for j in grid:
        start = [i*10, j*10]
        
        # Máximo Descenso
        _, iter_gd, _, val_gd, t_gd = maximo_descenso(start)
        results.append({'Algoritmo': 'Max Descenso', 'x_ini': start[0], 'Iter': iter_gd, 'Final_Val': val_gd, 'Tiempo': t_gd})
        
        # Newton
        _, iter_nm, _, val_nm, t_nm = newton_method(start)
        results.append({'Algoritmo': 'Newton', 'x_ini': start[0], 'Iter': iter_nm, 'Final_Val': val_nm, 'Tiempo': t_nm})
        
        # Quasi-Newton
        _, iter_qn, _, val_qn, t_qn = quasi_newton_bfgs(start)
        results.append({'Algoritmo': 'Quasi-Newton', 'x_ini': start[0], 'Iter': iter_qn, 'Final_Val': val_qn, 'Tiempo': t_qn})

df = pd.DataFrame(results)
print(df.groupby('Algoritmo')[['Iter', 'Tiempo', 'Final_Val']].mean())

# --- 4. Gráfica de Trayectorias (Clave para el informe) ---
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
Z = np.exp(X) * np.arctan(X**2 + Y**2)
plt.contour(X, Y, Z, levels=30, cmap='gray', alpha=0.4)

p0 = [10, 10] # Un punto "fácil"
p1 = [-5, 5]  # Un punto "difícil"

for p, style in zip([p0, p1], ['-', '--']):
    _, _, path_gd, _, _ = maximo_descenso(p)
    path_gd = np.array(path_gd)
    plt.plot(path_gd[:,0], path_gd[:,1], 'r', linestyle=style, label=f'Max Descenso {p}')
    
    _, _, path_qn, _, _ = quasi_newton_bfgs(p)
    path_qn = np.array(path_qn)
    plt.plot(path_qn[:,0], path_qn[:,1], 'b', linestyle=style, label=f'Quasi-Newton {p}')

plt.legend()
plt.title("Comparación de Trayectorias: Máximo Descenso vs Quasi-Newton")
plt.xlim(-20, 20); plt.ylim(-20, 20)
plt.savefig("trayectorias_optimas.png")
plt.show()