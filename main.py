from comunicacionArduino import send_command
from analisisMapa import execute_policy

# Configuración inicial
grid_size = (3, 3)
cell_width = 100
cell_height = 100
initial_position = (10, 10)

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

# Ejecutar la política
execute_policy(grid_size, cell_width, cell_height, policies, initial_position, send_command)
