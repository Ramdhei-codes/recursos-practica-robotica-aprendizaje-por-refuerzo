import numpy as np
from concurrent.futures import ThreadPoolExecutor

def inicializar_Q(nS, nA, tipo='zeros'):
    """
    Inicializa la tabla Q(S, A) con valores iniciales.

    Args:
        nS: Número de estados.
        nA: Número de acciones.
        tipo: Tipo de inicialización ('zeros' o 'random').

    Returns:
        np.ndarray: Tabla Q inicializada.
    """
    if tipo == 'zeros':
        return np.zeros((nS, nA))
    elif tipo == 'random':
        return np.random.rand(nS, nA)


def e_greedy(s, Q, epsilon, nA):
    """
    Selecciona una acción usando la política epsilon-greedy.

    Args:
        s: Estado actual.
        Q: Tabla Q.
        epsilon: Probabilidad de explorar.
        nA: Número de acciones.

    Returns:
        int: Acción seleccionada.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, nA)  # Acción aleatoria
    return np.argmax(Q[s])  # Mejor acción según Q


def sarsa(env, alpha, gamma, epsilon, nS, nA, max_episodios):
    """
    Algoritmo SARSA para aprendizaje por refuerzo.

    Args:
        env: Entorno.
        alpha: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon: Probabilidad de explorar.
        nS: Número de estados.
        nA: Número de acciones.
        max_episodios: Número máximo de episodios.

    Returns:
        np.ndarray: Tabla Q entrenada.
        list: Retornos acumulados por episodio.
    """
    Q = inicializar_Q(nS, nA)
    retornos = []

    for episodio in range(max_episodios):
        s = env.reset()
        a = e_greedy(s, Q, epsilon, nA)
        retorno_acumulado = 0
        done = False

        while not done:
            s_, r, done, _ = env.step(a)
            a_ = e_greedy(s_, Q, epsilon, nA)
            retorno_acumulado += r

            # Actualización SARSA
            Q[s, a] += alpha * (r + gamma * Q[s_, a_] * (not done) - Q[s, a])

            s, a = s_, a_

        retornos.append(retorno_acumulado)

    return Q, retornos


def qlearning(env, alpha, gamma, epsilon, nS, nA, max_episodios):
    """
    Algoritmo Q-learning para aprendizaje por refuerzo.

    Args:
        env: Entorno.
        alpha: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon: Probabilidad de explorar.
        nS: Número de estados.
        nA: Número de acciones.
        max_episodios: Número máximo de episodios.

    Returns:
        np.ndarray: Tabla Q entrenada.
        list: Retornos acumulados por episodio.
    """
    Q = inicializar_Q(nS, nA)
    retornos = []

    for episodio in range(max_episodios):
        s = env.reset()
        retorno_acumulado = 0
        done = False

        while not done:
            a = e_greedy(s, Q, epsilon, nA)
            s_, r, done, _ = env.step(a)
            retorno_acumulado += r

            # Actualización Q-learning
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_]) * (not done) - Q[s, a])

            s = s_

        retornos.append(retorno_acumulado)

    return Q, retornos


def entrenar_paralelo(env, alpha, gamma, epsilon, nS, nA, max_episodios):
    """
    Entrena SARSA y Q-learning en paralelo.

    Args:
        env: Entorno.
        alpha: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon: Probabilidad de explorar.
        nS: Número de estados.
        nA: Número de acciones.
        max_episodios: Número máximo de episodios.

    Returns:
        dict: Resultados de ambos algoritmos.
    """
    resultados = {}

    def ejecutar_sarsa():
        return sarsa(env, alpha, gamma, epsilon, nS, nA, max_episodios)

    def ejecutar_qlearning():
        return qlearning(env, alpha, gamma, epsilon, nS, nA, max_episodios)

    with ThreadPoolExecutor() as executor:
        futuros = {
            "SARSA": executor.submit(ejecutar_sarsa),
            "Q-learning": executor.submit(ejecutar_qlearning)
        }

        for nombre, futuro in futuros.items():
            resultados[nombre] = futuro.result()

    return resultados
