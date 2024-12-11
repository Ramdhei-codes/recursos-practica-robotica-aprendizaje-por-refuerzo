import serial
import time

# Confiqguración del puerto serial
PORT = "COM3"  # Cambia esto según el puerto asignado al Bluetooth
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
    print("verificando arduino")
    if bt_connection.is_open and bt_connection.in_waiting > 0:
        data = bt_connection.readline().decode('utf-8').strip()
        return data
    return None

# # Menú de control
# try:
#     while True:
#         print("\nControl del mBot:")
#         print("w - Adelante")
#         print("s - Atrás")
#         print("a - Izquierda")
#         print("d - Derecha")
#         print("x - Detener")
#         print("q - Salir")
#         command = input("Ingresa un comando: ").strip().lower()

#         if command == 'q':  # Salir del programa
#             print("Cerrando conexión...")
#             break
#         elif command in ['w', 's', 'a', 'd', 'x']:
#             send_command(command)  # Enviar el comando válido
#         else:
#             print("Comando no reconocido")

#         # Leer datos del Arduino y mostrarlos en la consola
#         response = read_from_arduino()
#         if response:
#             print(f"Arduino dice: {response}")

# except KeyboardInterrupt:
#     print("\nInterrupción por teclado.")

# finally:
#     # Cerrar la conexión al finalizar
#     if bt_connection.is_open:
#         bt_connection.close()
#         print("Conexión cerrada.")
