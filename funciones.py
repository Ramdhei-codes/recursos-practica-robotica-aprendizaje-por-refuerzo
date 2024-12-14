import numpy as np
import matplotlib.pyplot as plt

def mostrar_estado(env, S):
    '''Imprime en pantalla el estado actual (ubicación del taxi, 
    destino y origen)
    
    Parámetros:
    - env: el entorno de OpenAI Gym
    - S: el estado actual codificado
    '''
    
    # Decodificar el estado
    tx_fil, tx_col, origen, destino = env.env.decode(S)
    
    # Diccionarios
    origenes = {0: 'R(ed)', 1: 'G(reen)', 2: 'Y(ellow)', 3: 'B(lue)', 4:'In taxi'}
    destinos = {0: 'R(ed)', 1: 'G(reen)', 2: 'Y(ellow)', 3: 'B(lue)'}
    
    # Imprimir información en pantalla
    print(f'Ubicación del taxi (fil, col): {tx_fil+1, tx_col+1}')
    print(f'Origen: {origenes[origen]}')
    print(f'Destino: {destinos[destino]}')

# -----------------------------------------------------
def inicializar_Q(nS, nA, tipo='ones'):
    '''
    Inicializar la función Q(S,A) a valores iguales a 1.

    Parámetros:
    - nS: número de estados
    - nA: número de acciones

    Retorna la matriz Q(S,A)
    '''
    if tipo == 'ones':
    	Q = np.ones((nS,nA))
    elif tipo == 'random':
    	Q = np.random.rand(nS,nA)
    
    return Q

# -----------------------------------------------------
def e_greedy(s, Q, e, nA):
    '''
    Para un estado "s" selecciona una acción proveniente de Q(S,A) usando el enfoque e-greedy

    Parámetros:
    - s: el estado para el cual se desea obtener la acción
    - Q(S,A): tabla Q con S estados y A acciones
    - e: valor de épsilon (entre 0 y 1)
    - nA: número de acciones

    Retorna la acción seleccionada
    '''

    if np.random.rand() >= e:
        # Tomar la mejor acción con una probabilidad 1-e
        accion = np.argmax(Q[s, :])
    else:
        # Tomar cualquier acción aleatoriamente (con probabilidad e)
        accion = np.random.randint(0, nA)
    
    return accion

# -----------------------------------------------------
def sarsa(env, alpha, gamma, epsilon, nS, nA, K, verbose=False):
    '''
    Algoritmo de control SARSA

    Parámetros:
    - env: el entorno
    - alpha: la tasa de aprendizaje (entre 0 y 1)
    - gamma: factor de descuento (entre 0 y 1)
    - epsilon: parámetro para el algoritmo e-greedy (entre 0 y 1)
    - nS, nA: número de estados y de acciones
    - K: número de episodios a generar
    - verbose: imprimir en pantalla el progreso del entrenamiento (por defecto falso)
    '''

    # Inicializar Q(S,A) y almacenar retorno (la evolucion del entrenamiento)
    Q = inicializar_Q(nS, nA)
    retorno = []

    # Correr K episodios de interacción y estimar Q(S,A)
    for episodio in range(K):
        # Inicializar retorno acumulado
        retorno_acumulado = 0

        # Ir al primer estado del episodio
        s = env.reset()

        # Elegir acción usando e-greedy
        a = e_greedy(s, Q, epsilon, nA)

        done = False # Aún no estamos en un estado terminal

        while not done:
            # Iniciar interacciones y actualizar función Q
            s_, r_, done, info = env.step(a)
            a_ = e_greedy(s_, Q, epsilon, nA)
            retorno_acumulado += r_

            if not done:
                # Actualizar Q si no es un episodio terminal
                Q[s, a] += alpha * ( r_ + (gamma * Q[s_, a_] ) - Q[s, a] )
            else:
                # Si el episodio es terminal no se añade gamma * Q[S_, a_]
                # (porque Q[s_, a_] = 0 por ser estado terminal)
                Q[s, a] += alpha * ( r_ - Q[s, a] )
                retorno.append(retorno_acumulado)
        
            # Actualizar el estado y la acción
            s, a = s_, a_

            # Imprimir información cada 100 episodios
            if verbose:
                if episodio%100 == 0:
                    print(f'Episodio: {episodio+1}/{K}:')
                    print(f'\t\tRecompensa: {r_}')


    # Al finalizar tendremos la tabla Q, que implícitamente
    # contendrá la política
    return Q, retorno

# -----------------------------------------------------
def qlearning(env, alpha, gamma, epsilon, nS, nA, K, verbose=False):
    '''
    Algoritmo de control Q-learning

    Parámetros:
    - env: el entorno
    - alpha: la tasa de aprendizaje (entre 0 y 1)
    - gamma: factor de descuento (entre 0 y 1)
    - epsilon: parámetro para el algoritmo e-greedy
    - nS, nA: número de estados y de acciones
    - K: número de episodios a generar
    - verbose: imprimir en pantalla el progreso del entrenamiento (por defecto falso)
    '''

    # Inicializar Q(S,A) y almacenar retorno (la evolución del entrenamiento)
    Q = inicializar_Q(nS, nA)
    retorno = []

    # Correr K episodios de interacción y estimar Q(S,A)
    for episodio in range(K):
        # Inicializar retorno total
        retorno_acumulado = 0

        # Ir al primer estado del episodio
        s = env.reset()

        # Elegir acción usando e-greedy
        a = e_greedy(s, Q, epsilon, nA)

        done = False # Aún no estamos en un estado terminal

        while not done:
            # Iniciar interacciones y actualizar función Q
            s_, r_, done, info = env.step(a)
            a_ = np.argmax(Q[s_, :]) # <---- ESTA ES LA DIFERENCIA CON SARSA
            retorno_acumulado += r_

            if not done:
                # Si no es un episodio terminal, actualizar Q
                Q[s, a] += alpha * ( r_ + (gamma * Q[s_, a_] ) - Q[s, a] )
            else:
                # Si el episodio es terminal no se añade gamma * Q[S_, a_]
                # (porque Q[s_, a_] = 0 por ser estado terminal)
                Q[s, a] += alpha * ( r_ - Q[s, a] )
                retorno.append(retorno_acumulado)
        
            # Actualizar el estado y la acción
            s, a = s_, a_

            # Imprimir información cada 100 episodios
            if verbose:
                if episodio%100 == 0:
                    print(f'Episodio: {episodio+1}/{K}:')
                    print(f'\t\tRecompensa final: {r_}')


    # Al finalizar tendremos la tabla Q, que implícitamente
    # contendrá la política
    return Q, retorno

# -----------------------------------------------------
def interaccion(env, Q, method=None):
    it = 0
    s = env.reset()
    
    print('Resultado de la interacción:')
    print('-'*60)
    print('-'*60)
    
    # Imprimir estado y entorno iniciales
    print('**** Entorno-agente al inicio: ****')
    env.render()
    _, _, _, dest_i = env.env.decode(s)
    mostrar_estado(env, s)

    print('\nInteractuando...\n')
    
    # Hacer que el agente interactúe con el entorno tomando como política
    # el máximo de la función Q para cada estado
    a = np.argmax(Q[s,:]) # ¡La política!
    done = False

    # Interactuar hasta llegar a un estado terminal
    while not done:
        it += 1
        # Iniciar interacciones y actualizar función Q
        s, r, done, info = env.step(a)
        a = np.argmax(Q[s, :])
    
    # Imprimir información final
    _, _, _, dest_f = env.env.decode(s)
    if dest_i == dest_f:
        print(f'\nEl episodio culminó ***EXITOSAMENTE*** en {it} iteraciones')
    else:
        print(f'\nEl episodio culminó en {it} iteraciones')
    
    print('**** Entorno-agente al final: ****')
    env.render()
    mostrar_estado(env, s)
    print(f'Recompensa obtenida: {r}')

# -----------------------------------------------------
def graficar_entrenamiento(retornos, titulos):

    # Generar figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6),
    	sharex=True, sharey=True)
    
    # Graficar
    ax1.plot(retornos[0])
    ax1.title.set_text(titulos[0])
    ax2.plot(retornos[1], 'tab:orange')
    ax2.title.set_text(titulos[1])
    ax3.plot(retornos[2], 'tab:green')
    ax3.title.set_text(titulos[2])
    ax4.plot(retornos[3], 'tab:red')
    ax4.title.set_text(titulos[3])

    # Etiquetar ejes
    plt.setp([ax3, ax4], xlabel='Iteración')
    plt.setp([ax1, ax3], ylabel='Retorno')
