import cv2
import numpy as np
import random

# URL de DroidCam
url = "http://192.168.253.94:4747/video"

# Parámetros de la cuadrícula
rows = 7  # Número de filas
cols = 7  # Número de columnas
thickness = 1  # Grosor de las líneas

# Política para el mapa 3x3 como ejemplo
policies = {
    0: [0, 0, 0, 1],
    1: [0, 0, 0, 1],
    2: [0, 1, 0, 0],
    3: [0, 0, 1, 0],
    4: [0, 0, 1, 0],
    5: [0, 1, 0, 0],
    6: [0, 0, 0, 1],
    7: [0, 0, 0, 1],
    8: [0, 0, 0, 0]
}

def get_cell(x, y, cell_width, cell_height):
    """Devuelve la celda actual en base a las coordenadas cartesianas."""
    col = int(x // cell_width)
    row = int(y // cell_height)
    return row * cols + col  # Convierte a cell_index

def determine_direction(policy):
    """Devuelve la dirección según la política."""
    directions = ["up", "down", "left", "right"]
    for i, value in enumerate(policy):
        if value == 1:
            return directions[i]
    return None

def move_robot(direction):
    """Envía el comando para mover el robot."""
    if direction == "up":
        send_command('w')  # Comando para avanzar
    elif direction == "down":
        send_command('s')  # Comando para retroceder
    elif direction == "left":
        send_command('a')  # Comando para girar izquierda
    elif direction == "right":
        send_command('d')  # Comando para girar derecha

def send_command(command):
    """Ejemplo de función para enviar comandos al robot."""
    print(f"Enviando comando: {command}")
    # Aquí iría la lógica de comunicación serial para mover el robot

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

# Procesamiento de video
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo conectar a la cámara en la URL proporcionada.")
else:
    print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")

    cell_height = 480 // rows  # Altura de cada celda
    cell_width = 640 // cols  # Ancho de cada celda

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break

        # Obtener coordenadas del robot (aquí integrarías detección real)
        # Suponiendo que el robot está en el centro de la celda inicial
        x_robot, y_robot = 0, 0  # Esto debería calcularse dinámicamente

        # Obtener celda actual
        current_cell = get_cell(x_robot, y_robot, cell_width, cell_height)
        print(f"Celda actual: {current_cell}")

        # Consultar política de la celda
        if current_cell in policies:
            policy = policies[current_cell]
            print(f"Política para celda {current_cell}: {policy}")

            # Si la política es detenerse
            if policy == [0, 0, 0, 0]:
                print("Robot debe detenerse. Finalizando...")
                break

            # Determinar dirección y ejecutar movimiento
            direction = determine_direction(policy)
            while True:
                move_robot(direction)  # Mueve el robot
                # Simulación: Actualiza coordenadas (puedes usar datos reales aquí)
                x_robot += cell_width if direction == "right" else 0
                y_robot += cell_height if direction == "down" else 0

                # Verificar si cambió de celda
                new_cell = get_cell(x_robot, y_robot, cell_width, cell_height)
                if new_cell != current_cell:
                    print(f"Robot cambió a la celda {new_cell}")
                    break

        # Dibuja la cuadrícula y muestra el frame
        frame_with_grid = draw_grid(frame, rows, cols, thickness)
        cv2.imshow('Cuadrícula con análisis', frame_with_grid)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
