import numpy as np
from funciones import *

class PoliceEnvironment:
    def __init__(self, maze):
        self.actions = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        
        self.maze = maze 
        self.states = self.generar_estados_laberinto(maze)
        self.n = len(maze)
        self.m = len(maze[0])
        self.estado_actual = (0, 0, np.random.randint(0, (self.n * self.m) -1))
        self.role = 0
        self.info = ''
        self.state_to_index = {state: idx for idx, state in enumerate(self.states.keys())}
        self.index_to_state = {idx: state for state, idx in self.state_to_index.items()}
        self.movement_counter = 0


    def generar_estados_laberinto(self, laberinto):
        """
        Genera todos los estados posibles para un modelo de Policías y Ladrones
        y calcula las recompensas asociadas.

        Args:
            laberinto (list[list[int]]): Matriz n x n que representa el laberinto.

        Returns:
            dict: Diccionario donde las claves son los estados en formato (rol, pos_pol, pos_lad)
                y los valores son las recompensas asociadas.
        """
        n = len(laberinto)  # Tamaño del laberinto (asume que es cuadrado)
        num_casillas = n * n
        estados = {}

        def calcular_distancia_manhattan(pos1, pos2):
            x1, y1 = divmod(pos1, n)
            x2, y2 = divmod(pos2, n)
            return abs(x1 - x2) + abs(y1 - y2)

        for pos_pol in range(num_casillas):
            for pos_lad in range(num_casillas):
                for rol in [0, 1]:  # 0: Policía, 1: Ladrón
                    estado = (rol, pos_pol, pos_lad)

                    if rol == 0:  # Policía
                        if pos_pol == pos_lad:
                            recompensa = 100  # Captura al ladrón
                        else:
                            i, j = divmod(pos_pol, n)
                            recompensa = -1 - (- calcular_distancia_manhattan(pos_pol, pos_lad)) if laberinto[i][j] == 0 else -100
                    else:  # Ladrón
                        if pos_pol == pos_lad:
                            recompensa = -100  # Capturado por el policía
                        else:
                            recompensa = calcular_distancia_manhattan(pos_pol, pos_lad)

                    estados[estado] = recompensa
        print(estados)
        return estados
    
    def step(self, action):
        """
        Realiza una acción y actualiza el estado del entorno.

        Args:
            action (int): Acción a realizar (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            tuple: (nuevo_estado, recompensa, done)
        """
        self.movement_counter += 1
        rol, pos_pol, pos_lad = self.estado_actual
        filas, columnas = self.n, self.m
        nueva_pos = pos_pol if rol == 0 else pos_lad

        # Determinar nueva posición basada en la acción
        x, y = divmod(nueva_pos, columnas)

        if action == 0 and x > 0:  # UP
            x -= 1
        elif action == 1 and x < filas - 1:  # DOWN
            x += 1
        elif action == 2 and y > 0:  # LEFT
            y -= 1
        elif action == 3 and y < columnas - 1:  # RIGHT
            y += 1

        nueva_pos = x * columnas + y
        nuevo_estado = ()
        # Actualizar el estado basado en el rol
        if rol == 0:  # Policía
            nuevo_estado = (1, nueva_pos, pos_lad)
        else:  # Ladrón
            nuevo_estado = (0, pos_pol, nueva_pos)

        # Calcular la recompensa
        done = nuevo_estado[1] == nuevo_estado[2] or self.movement_counter == self.n*10
        recompensa = self.states[nuevo_estado]
        
        if self.estado_actual[0] == 0:
            recompensa -= self.movement_counter
        else:
            recompensa += self.movement_counter

        self.estado_actual = nuevo_estado
        return self.state_to_index[nuevo_estado], recompensa, done, self.info
    
    def reset(self):
        """
        Restaura el entorno al estado inicial.

        :return: int, el estado inicial
        """
        self.estado_actual = (0, 0, np.random.randint(0, (self.n * self.m) -1))
        self.movement_counter = 0
        return self.state_to_index[self.estado_actual]
    
    def generar_politica_policias(self, tabla_q):
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
            estado_tupla = self.index_to_state[estado]
            accion_maxima = np.argmax(tabla_q[estado])

            # Crear la lista de acciones con un 1 en la dirección de la acción máxima
            accion = [0] * n_acciones
            accion[accion_maxima] = 1

            # Asignar la política al estado actual
            policies[estado_tupla] = accion

        return policies
    

maze = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
model = PoliceEnvironment(maze)

# Hiperparámetros
ALPHA = 0.4
GAMMA = 0.999
EPSILON = 0.1 # (Probabilidad de escoger mejor acción = 1-0.1 = 0.9)
K = 2000

tabla_Q, _ret = sarsa(model, ALPHA, GAMMA, EPSILON, len(model.states.keys()), 4, K)

from utils import print_readable_dict
print(tabla_Q)
policy = model.generar_politica_policias(tabla_Q)

print_readable_dict(policy)


