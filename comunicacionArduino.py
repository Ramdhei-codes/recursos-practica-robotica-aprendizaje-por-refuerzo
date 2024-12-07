import serial
import time

# Configuración del puerto serial
PORT = "COM5"  # Cambia esto según el puerto asignado al Bluetooth
BAUD_RATE = 115200  # Velocidad de comunicación

# Crear la conexión serial
try:
    bt_connection = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Conectado al puerto {PORT}")
    time.sleep(2)  # Esperar a que el módulo Bluetooth esté listo
except Exception as e:
    print(f"Error al conectar con el puerto {PORT}: {e}")
    exit()

# Función para enviar un comando
def send_command(command):
    if bt_connection.is_open:
        bt_connection.write(command.encode('utf-8'))  # Enviar el comando como bytes
        print(f"Comando enviado: {command}")
        time.sleep(0.1)  # Pausa breve para evitar congestión
    else:
        print("La conexión serial no está abierta")

# Función para leer datos del Arduino
def read_from_arduino():
    if bt_connection.is_open and bt_connection.in_waiting > 0:
        data = bt_connection.readline().decode('utf-8').strip()
        return data
    return None

# Diccionario de estados
states = {
    0: [0, 0, 0, 1],  
    1: [0, 0, 0, 1], 
    2: [0, 1, 0, 0],
    3: [0, 1, 0, 0],
    4: [0, 0, 1, 0],
    5: [0, 0, 1, 0],
    6: [0, 0, 0, 1], 
    7: [0, 0, 0, 1],
    8: [0, 0, 0, 0], 
}

# Mapear direcciones al comando correspondiente
direction_commands = ['w', 's', 'd', 'a']  # arriba, abajo, derecha, izquierda

# Estado inicial
current_state = 0

try:
    while True:
        print(f"\nEstado actual: {current_state}")
        direction = states.get(current_state, [0, 0, 0, 0])  # Obtener la dirección para el estado actual
        if sum(direction) == 1:  # Validar que haya una sola dirección activa
            command_index = direction.index(1)
            command = direction_commands[command_index]
            send_command(command)  # Enviar el comando correspondiente

            # Leer datos del Arduino y mostrarlos en la consola
            response = read_from_arduino()
            if response:
                print(f"Arduino dice: {response}")
            
            # Avanzar al siguiente estado (puedes personalizar esta lógica)
            current_state = (current_state + 1) % len(states)
        else:
            print("Dirección inválida en el estado actual.")
            break

except KeyboardInterrupt:
    print("\nInterrupción por teclado.")

finally:
    # Cerrar la conexión al finalizar
    if bt_connection.is_open:
        bt_connection.close()
        print("Conexión cerrada.")
