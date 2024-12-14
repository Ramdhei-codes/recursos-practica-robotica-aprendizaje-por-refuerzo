import cv2
import numpy as np
import random
import comunicacionArduino
# URL de DroidCam
url = "http://192.168.128.189:4747/video"

# Parámetros de la cuadrícula
rows = 3  # Número de filas
cols = 3  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150

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
        random.shuffle(direcciones)  # Aleatoriza el orden de las direcciones
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
def draw_grid(frame, rows, cols, thickness=1):
    """Dibuja una cuadrícula en el frame."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    for i in range(1, rows):  # Líneas horizontales
        cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)
    for j in range(1, cols):  # Líneas verticales
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, height), (0, 255, 0), thickness)
    return frame


def detect_shapes_in_image(image, rows, cols, threshold1, threshold2,dilatacion):
    """Detecta círculos y triángulos en la imagen completa y calcula las celdas correspondientes."""
    detected_shapes = []
    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Umbral inverso para detectar regiones negras
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Detección de círculos con HoughCircles
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )

    # Procesar círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convertir a enteros
        for circle in circles:
            center_x, center_y, radius = circle
            row = center_y // cell_height
            col = center_x // cell_width
            cell_index = row * cols + col  # Índice de la celda
            cell_center_x = col * cell_width + cell_width // 2
            cell_center_y = row * cell_height + cell_height // 2

            # Dibujar círculo
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cell_index}",
                (center_x-10, center_y ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            cv2.putText(
                image,
                f"{center_x},{center_y}",
                (center_x - 30, center_y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            detected_shapes.append({
                "shape": "circle",
                "row": row,
                "col": col,
                "cell_index": cell_index,
                "x":center_x,
                "y":center_y,
                "cell_center_x":cell_center_x,
                "cell_center_y": cell_center_y,
            })
            image = draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height)

    bordes = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones((dilatacion, dilatacion), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes Modificado", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is not None:
        jerarquia = jerarquia[0]
    i=0
    for contour in figuras:
        if jerarquia[i][3] == -1:
            approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if len(approx) == 3 and area >= 500 and area < 3000:  # Triángulo
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2  # Centro aproximado del triángulo
                row = center_y // cell_height
                col = center_x // cell_width
                cell_index = row * cols + col  # Índice de la celda

                # Dibujar triángulo
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
                cv2.putText(
                    image,
                    f"{cell_index}",
                    (center_x , center_y+10 ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    image,
                    f"{center_x},{center_y}",
                    (center_x - 30, center_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                cell_center_x = col * cell_width + cell_width // 2
                cell_center_y = row * cell_height + cell_height // 2

                detected_shapes.append({
                    "shape": "triangle",
                    "row": row,
                    "col": col,
                    "cell_index": cell_index,
                    "x": center_x,
                    "y": center_y,
                    "cell_center_x": cell_center_x,
                    "cell_center_y": cell_center_y,
                })
                image=draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height)
                break
    return detected_shapes, image

def draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height):
    """Dibuja una línea punteada roja dentro de la celda en los ejes del centro de la celda."""
    # Definir los límites de la celda
    cell_left = cell_center_x - cell_width // 2
    cell_right = cell_center_x + cell_width // 2
    cell_top = cell_center_y - cell_height // 2
    cell_bottom = cell_center_y + cell_height // 2

    # Dibujar línea punteada roja en el eje horizontal
    for x in range(cell_left, cell_right, 10):  # Incremento para punteado
        cv2.line(image, (x, cell_center_y), (x + 5, cell_center_y), (0, 0, 255), 1)

    # Dibujar línea punteada roja en el eje vertical
    for y in range(cell_top, cell_bottom, 10):  # Incremento para punteado
        cv2.line(image, (cell_center_x, y), (cell_center_x, y + 5), (0, 0, 255), 1)
    return image
def highlight_start_end(frame, rows, cols):
    """Colorea en translúcido verde (0,0) y rojo (rows-1, cols-1)."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    # Coordenadas del inicio (0, 0)
    x1_start, y1_start = 0, 0
    x2_start, y2_start = cell_width, cell_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1_start, y1_start), (x2_start, y2_start), (0, 255, 0), -1)  # Verde

    # Coordenadas del final (rows-1, cols-1)
    x1_end, y1_end = (cols - 1) * cell_width, (rows - 1) * cell_height
    x2_end, y2_end = x1_end + cell_width, y1_end + cell_height
    cv2.rectangle(overlay, (x1_end, y1_end), (x2_end, y2_end), (0, 0, 255), -1)  # Rojo

    # Agregar transparencia
    alpha = 0.5  # Nivel de transparencia
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def on_trackbar_change(x):
    """Callback para manejar los cambios en las trackbars."""
    pass
        
    
def fill_cells(frame, matrix, alpha=0.7):
    """Rellena de color negro translúcido los cuadrantes correspondientes a los valores '1' en la matriz."""
    rows, cols = len(matrix), len(matrix[0])
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    overlay = frame.copy()  # Hacemos una copia para aplicar el color translúcido

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                # Coordenadas del cuadrante
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                # Rellenar el cuadrante con color negro (translúcido)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Aplicar transparencia a los rectángulos negros
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
def move_robot(pos_info, policy, prev_move):
    prev_index = prev_move['cell_index']
    index = pos_info['cell_index']
    if index != prev_index:
        pass
    else:
        comunicacionArduino.send_command('w')
    return pos_info

# Abre el video desde la URL
cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo conectar a la cámara en la URL proporcionada.")
else:
    print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, on_trackbar_change)
    cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, on_trackbar_change)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, on_trackbar_change)
    #maze = maze_generate(rows, cols)
    maze=[[0,0,0], [0,0,0], [0,0,0]]
        # Política del mapa
    policies = {
        0: [0, 0, 0, 1],  # Moverse a la derecha
        1: [0, 0, 0, 1],  # Moverse a la derecha
        2: [0, 1, 0, 0],  # Moverse abajo
        3: [0, 0, 1, 0],  # Moverse a la izquierda
        4: [0, 0, 1, 0],  # Moverse a la izquierda
        5: [0, 1, 0, 0],  # Moverse abajo
        6: [0, 0, 0, 1],  # Moverse a la derecha
        7: [0, 0, 0, 1],  # Moverse a la derecha
        8: [0, 0, 0, 0],  # Detenerse
    }
    #q_table = q_learning(maze)
    i = 0
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
        detected_shapes, frame_with_shapes = detect_shapes_in_image(frame, rows, cols, threshold1, threshold2,dilatacion)
        print(detected_shapes)
        if i % 25 == 0:
            previous_move = 0
            if len(detected_shapes) > 0:
                previous_move = move_robot(detected_shapes[0], policies, previous_move)
        i += 1
        # Dibujar la cuadrícula en el frame
        frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)

        fill_cells(frame_with_grid,maze)
        frame = highlight_start_end(frame, rows, cols)
        # Mostrar el frame con los ajustes
        cv2.imshow('Cuadrícula con análisis', frame_with_grid)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# Al final de analisisMapa.py

def get_next_direction(policy):
    """
    Retorna el movimiento basado en la política.
    Orden: [arriba, abajo, izquierda, derecha]
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Movimientos relativos
    for i, move in enumerate(policy):
        if move == 1:
            return directions[i]
    return None  # Si no hay movimiento, retorna None

def execute_policy(grid_size, cell_width, cell_height, policies, initial_position, send_command):
    """
    Ejecuta los movimientos del robot en un mapa basado en una política.
    """
    current_x, current_y = initial_position  # Posición inicial del robot
    rows, cols = grid_size
    cell_index = lambda x, y: (y // cell_height) * cols + (x // cell_width)

    while True:
        # Obtener la celda actual
        current_cell_index = cell_index(current_x, current_y)

        # Consultar la política de la celda actual
        if current_cell_index not in policies:
            print(f"No hay política definida para la celda {current_cell_index}.")
            break

        policy = policies[current_cell_index]

        # Detenerse si la política es [0, 0, 0, 0]
        if policy == [0, 0, 0, 0]:
            print(f"El robot se detiene en la celda {current_cell_index}.")
            send_command("x")  # Enviar comando de detener
            break

        # Obtener la dirección del próximo movimiento
        direction = get_next_direction(policy)
        if direction is None:
            print(f"No hay movimiento definido en la celda {current_cell_index}.")
            break

        dx, dy = direction

        # Simular movimiento
        print(f"Ejecutando movimiento desde celda {current_cell_index} en dirección {direction}.")

        # Enviar comando al robot
        if dx == -1:  # Arriba
            send_command("w")
        elif dx == 1:  # Abajo
            send_command("s")
        elif dy == -1:  # Izquierda
            send_command("a")
        elif dy == 1:  # Derecha
            send_command("d")

        # Actualizar posición simulada
        next_x = current_x + dx * cell_width // 4  # Paso ajustado
        next_y = current_y + dy * cell_height // 4

        # Verificar si se cambió de celda
        new_cell_index = cell_index(next_x, next_y)
        if new_cell_index != current_cell_index:
            print(f"Celda cambiada: {current_cell_index} -> {new_cell_index}")
            current_x, current_y = next_x, next_y
        else:
            print(f"No se cambió de celda. Repitiendo movimiento...")
            current_x, current_y = next_x, next_y


# Libera recursos
cap.release()
cv2.destroyAllWindows()
