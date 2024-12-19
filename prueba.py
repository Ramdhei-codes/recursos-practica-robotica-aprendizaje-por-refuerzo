import numpy as np
import random
import itertools

# Parámetros globales
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Factor de exploración

# Acciones posibles
actions = ['up', 'down', 'left', 'right']

def crear_entorno(filas, columnas, obstaculos=None):
    """Crea un entorno con dimensiones variables y posibles obstáculos."""
    entorno = np.zeros((filas, columnas), dtype=int)
    
    # Añadir obstáculos si se proporcionan
    if obstaculos:
        for obs in obstaculos:
            entorno[obs[0], obs[1]] = 1
    
    return entorno

def generate_start_goal_combinations(filas, columnas):
    """Genera todas las combinaciones posibles de inicio y meta."""
    posiciones = list(itertools.product(range(filas), range(columnas)))
    
    # Filtrar para evitar que inicio y meta sean la misma posición
    combinaciones = []
    for inicio in posiciones:
        for meta in posiciones:
            if inicio != meta:
                combinaciones.append((inicio, meta))
    
    return combinaciones

def initialize_q_table(env):
    """Inicializa una tabla Q con ceros."""
    estados = env.shape[0] * env.shape[1]
    estadosTotales = estados * estados
    filas = estadosTotales * 2
    return np.zeros((filas, len(actions)))


def state_to_index(state, env):
    """
    Convierte un estado (x, y) en un índice lineal para la tabla Q.
    Devuelve valores secuenciales de 0 en adelante.
    """
    x, y = state
    return x * env.shape[1] + y  # Cálculo directo sin multiplicaciones adicionales


def choose_action(q_table, state, epsilon, env):
    """Elige una acción usando epsilon-greedy."""
    state_index = state_to_index(state, env)
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Explorar
    else:
        return np.argmax(q_table[state_index])  # Explotar

def get_next_state(state, action, env):
    """Calcula el siguiente estado basado en la acción."""
    x, y = state
    if action == 0:  # up
        x = max(0, x - 1)
    elif action == 1:  # down
        x = min(env.shape[0] - 1, x + 1)
    elif action == 2:  # left
        y = max(0, y - 1)
    elif action == 3:  # right
        y = min(env.shape[1] - 1, y + 1)
    return (x, y)

def get_reward(state, goal,policia=None):
    """Devuelve la recompensa asociada a un estado."""
    
    if policia:
        distancia_ladron_policia = np.sqrt((state[0] - policia[0])*2 + (state[1] - policia[1])*2)
        if distancia_ladron_policia < 2:  # Penaliza mucho si está muy cerca
            return -100
    if state == goal:
        return 100  # Meta alcanzada
    return -1  # Recompensa neutral para otras celdas

def is_terminal(state, goal):
    """Verifica si el estado es terminal (meta)."""
    return state == goal

def q_learning(num_episodes, env, inicio, goal,policia=None):
    """Implementación del algoritmo Q-Learning."""
    q_table = initialize_q_table(env)

    for episode in range(num_episodes):
        state = inicio  # Estado inicial

        while not is_terminal(state, goal):
            state_index = state_to_index(state, env)
            action = choose_action(q_table, state, epsilon, env)
            next_state = get_next_state(state, action, env)
            
            # Si la próxima posición tiene un obstáculo, no se mueve
            if env[next_state[0], next_state[1]] == 1:
                next_state = state

            reward = get_reward(next_state, goal,policia)

            next_state_index = state_to_index(next_state, env)

            # Actualización de Q-Learning
            q_table[state_index, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action]
            )

            state = next_state

    return q_table

def sarsa(num_episodes, env, inicio, goal,policia=None):
    """Implementación del algoritmo SARSA."""
    q_table = initialize_q_table(env)
    
    for episode in range(num_episodes):
        state = inicio  # Posición inicial 
        action = choose_action(q_table, state, epsilon, env)

        while not is_terminal(state, goal):
            state_index = state_to_index(state, env)
            next_state = get_next_state(state, action, env)
            
            # Si la próxima posición tiene un obstáculo, no se mueve
            if env[next_state[0], next_state[1]] == 1:
                next_state = state

            reward = get_reward(next_state, goal,policia)
            next_action = choose_action(q_table, next_state, epsilon, env)

            next_state_index = state_to_index(next_state, env)

            # Actualización SARSA
            q_table[state_index, action] += alpha * (
                reward + gamma * q_table[next_state_index, next_action] - q_table[state_index, action]
            )

            state, action = next_state, next_action

            if is_terminal(state, goal):  
                break
    return q_table

def format_q_table(q_table, env):
    """Formatear la tabla Q para facilitar su lectura."""
    formatted_dict = {}
    estados = env.shape[0] * env.shape[1]
    index = 0  # Contador de celdas

    for i in range(env.shape[0]):
        for j in range(env.shape[1]):
            state_index = i * env.shape[1] + j + estados * (i * env.shape[1] + j)
            # Obtén la acción con el valor máximo y construye la lista binaria
            max_action = np.argmax(q_table[state_index])
            action_list = [1 if k == max_action else 0 for k in range(len(actions))]

            # Agrega al diccionario con el índice de celda como clave
            formatted_dict[index] = action_list
            index += 1

    return formatted_dict

def calculate_safe_position(ladron, policia, entorno):
    filas, columnas = entorno.shape
    max_distancia = -1
    posicion_segura = ladron

    for x in range(filas):
        for y in range(columnas):
            if entorno[x, y] == 0:  # Solo considerar celdas transitables
                distancia = np.sqrt((x - policia[0])*2 + (y - policia[1])*2)
                # Penalizar celdas cercanas al policía
                penalizacion = -0.5 * (1 / (distancia + 1e-5))
                distancia += penalizacion
                if distancia > max_distancia:
                    max_distancia = distancia
                    posicion_segura = (x, y)

    return posicion_segura




def extract_initial_actions_only(num_episodes, entorno):
    """
    Ejecuta Q-Learning para cada combinación inicio-meta y retorna la acción en la posición inicial.
    Devuelve un diccionario con clave (0, IDInicio, IDMeta).
    """
    filas, columnas = entorno.shape
    combinaciones = generate_start_goal_combinations(filas, columnas)
    resultados = {}
    
    for inicio, goal in combinaciones:
        # Crear copia del entorno con la meta marcada
        entorno_local = entorno.copy()
        entorno_local[goal[0], goal[1]] = 2

        # Ejecutar Q-Learning para esta combinación
        q_table = q_learning(num_episodes, entorno_local, inicio, goal)
        
        # Obtener los índices del estado inicial y meta
        id_inicio = state_to_index(inicio, entorno_local)
        id_meta = state_to_index(goal, entorno_local)

        # Obtener la acción en la posición inicial
        acciones = q_table[id_inicio]
        
        # Guardar en el diccionario con clave (0, policia, ladron)
        resultados[(0, id_inicio, id_meta)] = list(acciones)
    
        posicion_segura = calculate_safe_position(inicio, goal, entorno)
        # Crear copia del entorno con la meta marcada
        entorno_local = entorno.copy()
        entorno_local[posicion_segura[0], posicion_segura[1]] = 2
        entorno_local[goal[0], goal[1]] = 3

        # Ejecutar Q-Learning para esta combinación
        q_table = q_learning(num_episodes, entorno_local, inicio, posicion_segura,goal)
        
        # Obtener los índices del estado inicial y meta
        id_inicio = state_to_index(inicio, entorno_local)
        id_meta = state_to_index(goal, entorno_local)

        # Obtener la acción en la posición inicial
        acciones = q_table[id_inicio]
        
        # Guardar en el diccionario con clave (0, ladron, policia)
        resultados[(1, id_inicio, id_meta)] = list(acciones)
    
    return resultados