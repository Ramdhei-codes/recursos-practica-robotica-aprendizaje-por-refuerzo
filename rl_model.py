from funciones import *

class RobotEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.actions = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        
        self.states = self.generar_tabla_estados(maze)
        self.current_position = (0, 0)  # Posición inicial del robot
        self.goal_position = (len(maze) - 1, len(maze[0]) - 1)
        self.info = ''
        
    def step(self, action):
        """
        Realiza una acción y actualiza el estado del entorno.

        :param action: int, la acción a realizar (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
        :return: (nuevo_estado, recompensa, done)
        """
        filas, columnas = len(self.maze), len(self.maze[0])
        x, y = self.current_position

        # Determinar nueva posición basada en la acción
        if action == 0 and x > 0:  # UP
            x -= 1
        elif action == 1 and x < filas - 1:  # DOWN
            x += 1
        elif action == 2 and y > 0:  # LEFT
            y -= 1
        elif action == 3 and y < columnas - 1:  # RIGHT
            y += 1

        # Actualizar la posición del robot
        self.current_position = (x, y)

        # Calcular el estado y la recompensa
        nuevo_estado = x * columnas + y
        recompensa = self.states[nuevo_estado]

        # Determinar si se ha alcanzado la meta
        done = self.current_position == self.goal_position

        return nuevo_estado, recompensa, done, self.info
    
    def generar_tabla_estados(self, matriz):
        """
        Genera una tabla de estados para un modelo de aprendizaje por refuerzo.

        Args:
            matriz (list[list[int]]): Matriz n x n que representa el entorno.

        Returns:
            dict: Diccionario donde las claves son las posiciones lineales (0 a n*n-1)
                y los valores son las recompensas asociadas.
        """
        n = len(matriz)  # Tamaño de la matriz (asume que es cuadrada)
        tabla_estados = {}

        for i in range(n):
            for j in range(n):
                # Calcular posición lineal a partir de índices (i, j)
                posicion = i * n + j

                # Asignar recompensas según el valor en la matriz
                if matriz[i][j] == 0:
                    recompensa = -1
                elif matriz[i][j] == 1:
                    recompensa = -20
                elif i == n - 1 and j == n - 1:  # Verificar si es la meta
                    recompensa = 30
                else:
                    recompensa = -1  # Recompensa por defecto si no se cumple ninguna condición

                # Guardar en la tabla de estados
                tabla_estados[posicion] = recompensa

        return tabla_estados
    
    def reset(self):
        """
        Restaura el entorno al estado inicial.

        :return: int, el estado inicial
        """
        self.current_position = (0, 0)
        filas, columnas = len(self.maze), len(self.maze[0])
        estado_inicial = self.current_position[0] * columnas + self.current_position[1] + 1
        return estado_inicial
    
class PoliceAndThievesEnvironment:
    def __init__(self, maze):
        self.maze = maze
        
    



def generar_politica(tabla_q):
    """
    Genera la política a partir de una tabla Q.

    Args:
        tabla_q (np.ndarray): Matriz Q con valores de estados y acciones.

    Returns:
        dict: Diccionario donde las claves son los estados y los valores son listas
              de acciones en formato [arriba, abajo, izquierda, derecha].
    """
    n_estados, n_acciones = tabla_q.shape
    policies = {}

    for estado in range(n_estados):
        # Obtener la acción con el valor máximo para el estado actual
        accion_maxima = np.argmax(tabla_q[estado])

        # Crear la lista de acciones con un 1 en la dirección de la acción máxima
        accion = [0] * n_acciones
        accion[accion_maxima] = 1

        # Asignar la política al estado actual
        policies[estado] = accion

    return policies

# Hiperparámetros
ALPHA = 0.4
GAMMA = 0.999
EPSILON = 0.1 # (Probabilidad de escoger mejor acción = 1-0.1 = 0.9)
K = 2000
maze = [[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]

model = RobotEnvironment(maze)

def select_algorithm(ALPHA, GAMMA, EPSILON, K, model, algorithm):
    if algorithm == 'sarsa':
        tabla_Q, ret = sarsa(model, ALPHA, GAMMA, EPSILON, len(model.states.keys()), 4, K)
        return generar_politica(tabla_Q)
    else:
        tabla_Q_learning, ret_q_learning = qlearning(model, ALPHA, GAMMA, EPSILON, len(model.states.keys()), 4, K)
        return generar_politica(tabla_Q_learning)
    
    
    


