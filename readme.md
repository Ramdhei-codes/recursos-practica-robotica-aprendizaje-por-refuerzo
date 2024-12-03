---

# Reto de Robótica Móvil y Aprendizaje Reforzado

Este proyecto utiliza **Python** para analizar un tablero mediante visión artificial y comunicarse con un robot a través de comandos seriales enviados a un Arduino. Está dividido en dos módulos principales: `analisisMapa` y `comunicacionArduino`.

## Requisitos previos

1. **Python 3.8+**  
2. **Entorno virtual** para gestionar las dependencias.
3. Instalación de dependencias desde el archivo `requirements.txt`.

## Instalación

1. **Clona este repositorio**:  
   ```bash
   git clone https://github.com/felipebuitragocarmona/recursos-practica-robotica-aprendizaje-por-refuerzo recursosRobticaAprendizajeReforzado
   cd recursosRobticaAprendizajeReforzado
   ```

2. **Crea y activa un entorno virtual**:  
   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/MacOS
   .\venv\Scripts\activate    # En Windows
   ```

3. **Instala las dependencias**:  
   ```bash
   pip install -r requirements.txt
   ```

## Estructura del proyecto

- `analisisMapa.py`: Este módulo utiliza técnicas de visión artificial para analizar un tablero.  
  - **Funcionalidad principal**:
    - Se debe configurar la IP de la cámara en la variable `url`
    - Detecta robots representados como círculos o triángulos.
    - Genera un laberinto de dimensiones `n x m`siempre de manera aleatoria.
    - Retorna las coordenadas de los robots detectados en el tablero.

- `comunicacionArduino.py`: Este módulo permite la comunicación serial con un robot controlado por un Arduino.  
  - **Comandos soportados**:
    - `w`: Mueve el robot hacia adelante.
    - `s`: Mueve el robot hacia atrás.
    - `d`: Gira o desplaza el robot hacia la derecha.
    - `a`: Gira o desplaza el robot hacia la izquierda.

## Uso del proyecto

1. **Ejecuta `analisisMapa.py`** para analizar el tablero:  
   ```bash
   python analisisMapa.py
   ```
   Este archivo genera un laberinto de dimensiones configurables y muestra las coordenadas de los robots detectados. Recuerde configurar la variable `url` para activar la cámara ip. La variable `maze` corresponde a la estructura del laberinto, donde 0 es camino y 1 es obstáculo. Recuerde que el robot debe verificar en el recorrido de exploración que ese mapa sea correcto. El robot debe pasar por cada una de las casillas en la fase de reconocimiento.

2. **Comunicación con Arduino**:  
   Asegúrate de que el Arduino esté conectado y configurado para recibir comandos seriales.  Para esto, primero debe agregar el dispositivo Bluetooth desde el sistema operativo, el robot se llama "MakeBlock". Luego verificar en el adiminstrador de dispositivos de Windows en la sección "Puertos COM y LPT) que puertos posibles hay conectados. Por último, dentro del script de comunicación cambie la variable "PORT".
   
   Ejecuta `comunicacionArduino.py` para enviar comandos:  
   ```bash
   python comunicacionArduino.py
   ```

   Ejemplo de uso interactivo:  
   - Ingresa los comandos `w`, `s`, `d`, o `a` según sea necesario.


