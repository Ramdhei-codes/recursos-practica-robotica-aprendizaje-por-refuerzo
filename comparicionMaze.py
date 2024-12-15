import time
import matplotlib
matplotlib.use('Agg')  # Cambia el backend para evitar problemas con Tk
import matplotlib.pyplot as plt

from rl_model import RobotEnvironment
from funciones import sarsa, qlearning

def comparar_algoritmos(env, alpha, gamma, epsilon, n_episodios, n_estados, n_acciones):
    """
    Compara SARSA y Q-learning en términos de velocidad de convergencia y tiempo de ejecución.

    Args:
        env: Entorno de simulación.
        alpha: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon: Parámetro épsilon-greedy.
        n_episodios: Número de episodios para el entrenamiento.
        n_estados: Número de estados.
        n_acciones: Número de acciones.

    Returns:
        dict: Contiene resultados de cada algoritmo (retornos y tiempos).
    """
    resultados = {}

    # SARSA
    start_time = time.time()
    Q_sarsa, retorno_sarsa = sarsa(env, alpha, gamma, epsilon, n_estados, n_acciones, n_episodios)
    tiempo_sarsa = time.time() - start_time
    resultados['SARSA'] = {'retornos': retorno_sarsa, 'tiempo': tiempo_sarsa}

    # Q-learning
    start_time = time.time()
    Q_qlearning, retorno_qlearning = qlearning(env, alpha, gamma, epsilon, n_estados, n_acciones, n_episodios)
    tiempo_qlearning = time.time() - start_time
    resultados['Q-learning'] = {'retornos': retorno_qlearning, 'tiempo': tiempo_qlearning}

    return resultados

def graficar_resultados(resultados, n_episodios, output_dir="graficos"):
    """
    Genera gráficos comparativos de los algoritmos SARSA y Q-learning.

    Args:
        resultados: Diccionario con los retornos y tiempos de los algoritmos.
        n_episodios: Número de episodios.
        output_dir: Directorio donde se guardarán los gráficos.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gráfico de retornos acumulados
    plt.figure()
    for algoritmo, datos in resultados.items():
        plt.plot(range(n_episodios), datos['retornos'], label=algoritmo)
    plt.xlabel("Episodios")
    plt.ylabel("Retorno acumulado")
    plt.title("Comparación de Retornos")
    plt.legend()
    plt.savefig(f"{output_dir}/retornos_comparacion.png")

    # Gráfico de tiempo de ejecución
    plt.figure()
    algoritmos = list(resultados.keys())
    tiempos = [datos['tiempo'] for datos in resultados.values()]
    plt.bar(algoritmos, tiempos, color=['blue', 'orange'])
    plt.xlabel("Algoritmo")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución")
    plt.savefig(f"{output_dir}/tiempos_comparacion.png")

    print(f"Gráficos guardados en {output_dir}")

# Configuración del entorno y parámetros
maze = [[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]
robot_env = RobotEnvironment(maze)

ALPHA = 0.4
GAMMA = 0.999
EPSILON = 0.1
N_EPISODIOS = 2000
N_ESTADOS = len(robot_env.states)
N_ACCIONES = 4

# Comparar algoritmos
resultados = comparar_algoritmos(robot_env, ALPHA, GAMMA, EPSILON, N_EPISODIOS, N_ESTADOS, N_ACCIONES)

# Graficar resultados
graficar_resultados(resultados, N_EPISODIOS)
