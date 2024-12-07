import cv2
import numpy as np
import random

# URL de DroidCam
url = "http://192.168.253.94:4747/video"

# Parámetros de la cuadrícula
rows = 3  # Número de filas
cols = 3  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150

# Dirección del robot: 0 = arriba, 1 = derecha, 2 = abajo, 3 = izquierda
robot_direction = 1  # Empezamos mirando hacia la derecha

def maze_generate(filas, columnas):
    """Genera un laberinto aleatorio con caminos (0) y paredes (1)."""
    maze = np.ones((filas, columnas), dtype=int)  # Inicializa el laberinto con paredes
    for i in range(filas):
        for j in range(columnas):
            if random.random() > 0.3:  # Aproximadamente el 70% de las celdas serán caminos
                maze[i][j] = 0  # 0 representa un camino
    maze[0][0] = 0  # Establece la casilla de inicio como un camino
    maze[filas-1][columnas-1] = 0  # Establece la casilla de fin como un camino
    return maze

def draw_grid(frame, rows, cols, thickness=1):
    """Dibuja la cuadrícula del laberinto en la imagen."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols
    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 0, 255), thickness)
    for j in range(1, cols):
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, height), (0, 0, 255), thickness)
    return frame

def detect_shapes_in_image(image, rows, cols, threshold1, threshold2, dilatacion):
    """Detecta las formas (caminos y paredes) en la imagen usando Canny y otros filtros."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones((dilatacion, dilatacion), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated, image

def fill_cells(frame, matrix, alpha=0.7):
    """Rellena las celdas del laberinto con colores según el estado del laberinto."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols
    for i in range(rows):
        for j in range(cols):
            color = (0, 255, 0) if matrix[i][j] == 0 else (0, 0, 255)  # Verde para caminos, rojo para paredes
            cv2.rectangle(frame, (j * cell_width, i * cell_height),
                          ((j + 1) * cell_width, (i + 1) * cell_height), color, -1)
    return frame

def highlight_start_end(frame, rows, cols):
    """Resalta el inicio y el final en el laberinto."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols
    cv2.circle(frame, (0 + cell_width // 2, 0 + cell_height // 2), 15, (255, 0, 0), -1)  # Inicio
    cv2.circle(frame, ((cols - 1) * cell_width + cell_width // 2, (rows - 1) * cell_height + cell_height // 2),
               15, (0, 0, 255), -1)  # Fin
    return frame

def move_robot(cell_index, direction, maze):
    """
    Mueve al robot según su dirección y valida que el movimiento sea correcto.
    La función valida si el movimiento lleva al robot a una casilla válida.
    """
    # Convertir el cell_index a coordenadas (fila, columna)
    row = cell_index // cols
    col = cell_index % cols
    
    # Direcciones de movimiento según la orientación actual del robot
    movement = {
        0: (-1, 0),  # Arriba
        1: (0, 1),   # Derecha
        2: (1, 0),   # Abajo
        3: (0, -1)   # Izquierda
    }

    # Coordenadas a las que el robot intentará moverse
    dx, dy = movement[direction]
    new_row = row + dx
    new_col = col + dy

    # Validar que la nueva casilla esté dentro del laberinto y sea un camino
    if 0 <= new_row < rows and 0 <= new_col < cols and maze[new_row][new_col] == 0:
        new_cell_index = new_row * cols + new_col
        return new_cell_index  # La nueva casilla es válida, movemos al robot
    return cell_index  # El movimiento no es válido, el robot permanece en la misma casilla

def execute_robot_commands(commands, current_cell_index, maze):
    """
    Ejecuta una lista de comandos para mover el robot.
    Los comandos pueden ser 'rotate_left', 'rotate_right' o 'move_forward'.
    """
    global robot_direction  # Usamos la variable global para la dirección del robot

    for command in commands:
        if command == 'rotate_left':
            robot_direction = (robot_direction - 1) % 4  # Rotar 90 grados a la izquierda
        elif command == 'rotate_right':
            robot_direction = (robot_direction + 1) % 4  # Rotar 90 grados a la derecha
        elif command == 'move_forward':
            # Validar si el movimiento hacia adelante es posible
            current_cell_index = move_robot(current_cell_index, robot_direction, maze)

    return current_cell_index

def on_trackbar_change(x):
    """Callback para manejar los cambios en las trackbars."""
    pass

# Abre el video desde la URL
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("No se pudo conectar a la cámara en la URL proporcionada.")
else:
    print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, on_trackbar_change)
    cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, on_trackbar_change)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, on_trackbar_change)
    maze = maze_generate(rows, cols)
    print(maze)
    
    current_cell_index = 0  # El robot comienza en la casilla de inicio (0,0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break

        # Obtener valores de las trackbars
        threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
        threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
        dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

        # Analizar el frame con los umbrales ajustados
        detected_shapes, frame_with_shapes = detect_shapes_in_image(frame, rows, cols, threshold1, threshold2, dilatacion)
        print(detected_shapes)
        
        # Ejecutar comandos de movimiento
        # Ejemplo: Ejecutar rotación, avance y otra rotación
        commands = ['rotate_right', 'move_forward', 'rotate_left', 'move_forward']
        current_cell_index = execute_robot_commands(commands, current_cell_index, maze)
        
        # Dibujar la cuadrícula en el frame
        frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)

        frame = fill_cells(frame_with_grid, maze)
        frame = highlight_start_end(frame, rows, cols)
        
        # Mostrar el frame con los ajustes
        cv2.imshow('Cuadrícula con análisis', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
