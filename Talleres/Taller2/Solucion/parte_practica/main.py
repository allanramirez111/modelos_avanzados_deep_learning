from number_data_loader import load_data
from network import Network
import numpy as np
from plot import plot_digits, plot_predicted_digits

# Función para cargar los datos
def load_df():
    """
    Carga los datos de entrenamiento y prueba del dataset MNIST y grafica ejemplos de dígitos.
    
    Retorna:
    - training_data (list): Lista de tuplas (x, y) para entrenamiento.
    - test_data (list): Lista de tuplas (x, y) para prueba.
    
    Los datos son imágenes de dígitos (x) y sus respectivas etiquetas (y).
    """
    # Cargar los datos utilizando la función de number_data_loader
    training_data, test_data = load_data()
    
    # Graficar ejemplos de los dígitos de entrenamiento
    plot_digits(training_data)
    
    return training_data, test_data  # Retornar los conjuntos de datos

# Función para entrenar la red neuronal
def entrenar_red(dict_config, training_data, test_data):
    """
    Entrena una red neuronal usando el algoritmo de gradiente descendente estocástico (SGD).

    Parámetros:
    - dict_config (dict): Diccionario con la configuración de la red neuronal (arquitectura, épocas, tamaño de mini-lotes y tasa de aprendizaje).
    - training_data (list): Datos de entrenamiento (imágenes y etiquetas).
    - test_data (list): Datos de prueba para evaluar la red durante el entrenamiento.
    
    Retorna:
    - net (Network): La red neuronal entrenada.
    """
    # Extraer la configuración del diccionario
    architecture = dict_config['architecture']  # Arquitectura de la red (número de neuronas por capa)
    epochs = dict_config['epochs']  # Número de épocas
    mini_batch_size = dict_config['mini_batch_size']  # Tamaño de mini-lote
    eta = dict_config['eta']  # Tasa de aprendizaje

    # Crear una instancia de la red neuronal con la arquitectura especificada
    net = Network(architecture)
    
    # Entrenar la red utilizando el algoritmo SGD
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
    
    # Añadir red a diccionario
    dict_config['red'] = net
    
    return net

# Función para medir el desempeño de la red neuronal
def medir_desempeno(net, test_data):
    """
    Mide el desempeño de la red neuronal calculando el porcentaje de aciertos en el conjunto de prueba.
    
    Parámetros:
    - net (Network): La red neuronal entrenada.
    - test_data (list): Lista de tuplas (x, y) donde x es la imagen y y es la etiqueta correcta (no en formato one-hot).
    
    Retorna:
    - precision (float): Porcentaje de predicciones correctas.
    """
    # Contador de predicciones correctas
    correct_predictions = 0
    
    for x, y in test_data:
        # Realizar predicción
        predicted_label = np.argmax(net.feedforward(x))  # Obtener el dígito predicho
        if predicted_label == y:
            correct_predictions += 1  # Incrementar el contador si es correcto
    
    # Calcular el porcentaje de aciertos
    precision = (correct_predictions / len(test_data)) * 100
    
    return precision

def entrenar_y_medir(config, training_data, test_data):
    """
    Función auxiliar que entrena la red y mide el desempeño en base a la configuración.
    
    Parámetros:
    - config (dict): Diccionario con los parámetros de configuración de la red.
    - training_data (list): Datos de entrenamiento (imágenes y etiquetas).
    - test_data (list): Datos de prueba para medir el desempeño de la red.
    
    Retorna:
    - config (dict): Diccionario con la configuración actualizada con la precisión medida.
    """
    # Entrenar la red
    net = entrenar_red(config, training_data, test_data)
    
    # Medir el desempeño
    precision = medir_desempeno(net, test_data)
    
    # Guardar la precisión en la configuración
    config['precision'] = precision
    print(f"Precisión de la red: {precision:.2f}%")
    
    return net, precision

# Función principal
def main():
    """
    Función principal que ejecuta el flujo del programa: carga los datos, los grafica, entrena la red y devuelve la red entrenada.
    
    La configuración por defecto incluye una red neuronal con una capa oculta de 30 neuronas,
    entrenada durante 30 épocas con una tasa de aprendizaje de 0.1.
    """
    # Configuración por defecto de la red neuronal
    config = {
        'architecture': [784, 30, 10],  # 784 neuronas en la entrada, 30 en la capa oculta y 10 en la salida
        'epochs': 30,  # Número de épocas de entrenamiento
        'mini_batch_size': 10,  # Tamaño de los mini-lotes
        'eta': 0.1  # Tasa de aprendizaje
    }

    # Cargar los datos de entrenamiento y prueba
    training_data, test_data = load_df()

    # Entrenar la red neuronal
    net = entrenar_y_medir(config, training_data, test_data)
    
    # Graficar resultados
    plot_predicted_digits(net, test_data)
    
    # Imprimmir precisión
    print(f"Precisión de la red: {config['precision']:.2f}%")
    
    return

# Ejecución principal
if __name__ == "__main__":
    # Ejecutar la función principal
    main()