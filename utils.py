import numpy as np
def maze_generate(filas, columnas):
    """
    Genera un laberinto de dimensiones filas x columnas.
    Los caminos están representados por 0 y las paredes por 1.
    Garantiza que (0,0) es el inicio y (filas-1,columnas-1) es la meta con un camino solucionable.
    """
    # Crear una matriz llena de paredes (1)
    laberinto = [[1 for _ in range(columnas)] for _ in range(filas)]

    # Direcciones de movimiento: (dx, dy) para celdas ortogonales
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def en_rango(x, y):
        """Verifica si una celda está dentro del rango del laberinto."""
        return 0 <= x < filas and 0 <= y < columnas

    def dfs(x, y):
        """Algoritmo DFS para construir el laberinto."""
        laberinto[x][y] = 0  # Marca el camino actual como "camino"
        np.random.shuffle(direcciones)  # Aleatoriza el orden de las direcciones
        for dx, dy in direcciones:
            nx, ny = x + 2 * dx, y + 2 * dy  # Saltar una celda para garantizar paredes entre caminos
            if en_rango(nx, ny) and laberinto[nx][ny] == 1:  # Si es una celda válida y no visitada
                # Romper la pared entre la celda actual y la siguiente
                laberinto[x + dx][y + dy] = 0
                # Continuar el DFS desde la celda siguiente
                dfs(nx, ny)

    # Inicializar el laberinto
    laberinto[0][0] = 0  # Crear la entrada
    dfs(0, 0)

    # Crear la salida
    laberinto[filas - 1][columnas - 1] = 0  # Asegurar que el punto final sea siempre un camino

    # Conectar la salida al camino más cercano si está aislada
    if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
        laberinto[filas - 2][columnas - 1] = 0  # Romper la pared superior

    # Devolver la matriz del laberinto
    return laberinto

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

# Ejemplo de uso
q_table = np.array([
    [-24.66475568, -29.97016698, -21.82413374, -16.99331734],
    [-19.37014749, -26.1590233 , -20.55935235,  -8.35964402],
    [-15.07909812,  -4.78466206, -14.97703673, -33.76886858],
    [-29.64873101, -21.02411532, -12.85456171, -30.72179646],
    [-13.2498475 , -10.59013128, -11.80064   , -18.24980518],
    [-10.56215477, -13.8970815 , -16.36232595, -10.67082003],
    [-15.94262051,  -3.34979135, -26.65199242, -22.03287819],
    [-24.08416184,  -1.99960809,  -7.24492724, -11.96063984],
    [-21.60333282, -20.15550791, -11.9365564 ,  -5.43737013],
    [-26.85098439, -21.54047142,  -9.1857138 ,  -6.9091145 ],
    [ -7.97536114, -10.65307833,  -8.61782641,  -2.09013988],
    [-23.90640844,  -1.        ,  -5.19858293,  -1.99898947],
    [ -5.91933952, -11.80064   , -11.80064   ,  -7.0004    ],
    [ -4.07147393, -11.80064   , -17.96384738,  -1.99899943],
    [ -4.37970084,  -2.05002159, -22.00259195,  -1.        ],
    [  1.        ,   1.        ,   1.        ,   1.        ]
])



def generar_estados_laberinto(laberinto):
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
                        recompensa = 50  # Captura al ladrón
                    else:
                        i, j = divmod(pos_pol, n)
                        recompensa = -1 if laberinto[i][j] == 0 else -20
                else:  # Ladrón
                    if pos_pol == pos_lad:
                        recompensa = -50  # Capturado por el policía
                    else:
                        recompensa = calcular_distancia_manhattan(pos_pol, pos_lad)

                estados[estado] = recompensa

    return estados

maze = maze_generate(3,3)
print(maze)
estados = generar_estados_laberinto(maze)
print(estados)
print(len(estados))

