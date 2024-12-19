import time
import matplotlib
matplotlib.use('Agg')  # Cambia el backend para evitar problemas con Tk
import matplotlib.pyplot as plt
from police_model import PoliceEnvironment, select_algorithm_policies
from funciones2 import sarsa, qlearning
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def comparar_algoritmos_policias(env, alpha, gamma, epsilon, n_episodios, n_estados, n_acciones):
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

    def entrenar_sarsa():
        start_time = time.time()
        Q_sarsa, retorno_sarsa = sarsa(env, alpha, gamma, epsilon, n_estados, n_acciones, n_episodios)
        tiempo_sarsa = time.time() - start_time
        return {'retornos': retorno_sarsa, 'tiempo': tiempo_sarsa}

    def entrenar_qlearning():
        start_time = time.time()
        Q_qlearning, retorno_qlearning = qlearning(env, alpha, gamma, epsilon, n_estados, n_acciones, n_episodios)
        tiempo_qlearning = time.time() - start_time
        return {'retornos': retorno_qlearning, 'tiempo': tiempo_qlearning}

    # Paralelizar entrenamiento
    with ThreadPoolExecutor() as executor:
        sarsa_future = executor.submit(entrenar_sarsa)
        qlearning_future = executor.submit(entrenar_qlearning)
        resultados['SARSA'] = sarsa_future.result()
        resultados['Q-learning'] = qlearning_future.result()

    return resultados

def graficar_resultados_policias(resultados, n_episodios, output_dir="graficos_policias"):
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
    plt.title("Comparación de Retornos (Policías y Ladrones)")
    plt.legend()
    plt.savefig(f"{output_dir}/retornos_comparacion_policias.png")

    # Gráfico de tiempo de ejecución
    plt.figure()
    algoritmos = list(resultados.keys())
    tiempos = [datos['tiempo'] for datos in resultados.values()]
    plt.bar(algoritmos, tiempos, color=['blue', 'orange'])
    plt.xlabel("Algoritmo")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución (Policías y Ladrones)")
    plt.savefig(f"{output_dir}/tiempos_comparacion_policias.png")

    print(f"Gráficos guardados en {output_dir}")

# Configuración del entorno y parámetros
maze = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
police_env = PoliceEnvironment(maze)

ALPHA = 0.4
GAMMA = 0.95  # Reducido para acelerar la convergencia
EPSILON = 0.05  # Menor exploración aleatoria
N_EPISODIOS = 2000  # Reducido para pruebas más rápidas
N_ESTADOS = len(police_env.states)
N_ACCIONES = 4

# Comparar algoritmos
resultados_policias = comparar_algoritmos_policias(police_env, ALPHA, GAMMA, EPSILON, N_EPISODIOS, N_ESTADOS, N_ACCIONES)

# Graficar resultados
graficar_resultados_policias(resultados_policias, N_EPISODIOS)
