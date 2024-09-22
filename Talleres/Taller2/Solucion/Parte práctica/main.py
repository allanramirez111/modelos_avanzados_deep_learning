from number_data_loader import load_data
from network import Network
import numpy as np
import matplotlib.pyplot as plt

# Función para gráfica de dígitos
def plot_digits(data, num_per_class=5):
    """
    Grafica una cantidad específica de dígitos por clase.

    Parámetros:
    - data (list of tuples): Lista de tuplas (x, y) donde x es la imagen y y es la etiqueta.
    - num_per_class (int): Número de ejemplos a graficar por clase.
    """
    # Crear un diccionario para almacenar los ejemplos por clase
    examples = {}
    for x, y in data:
        if y not in examples:
            examples[y] = []
        if len(examples[y]) < num_per_class:
            examples[y].append(x)
        if all(len(v) >= num_per_class for v in examples.values()):
            break

    # Determinar el número de clases
    classes = sorted(examples.keys())
    num_classes = len(classes)
    plt.figure(figsize=(num_classes * 2, num_per_class * 2))

    for idx, cls in enumerate(classes):
        for i in range(num_per_class):
            plt_idx = idx * num_per_class + i + 1
            plt.subplot(num_classes, num_per_class, plt_idx)
            img = examples[cls][i].reshape(28, 28)  # Asumiendo que las imágenes son 28x28
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.ylabel(str(cls), fontsize=12)
    plt.suptitle('Ejemplos de Dígitos de Entrenamiento', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Main
def main(dict_config):
    # Extraer la arquitectura de la red neuronal desde el diccionario de configuración
    architecture = dict_config['architecture']
    epochs = dict_config['epochs']
    mini_batch_size = dict_config['mini_batch_size']
    eta = dict_config['eta']
    
    # Cargar los datos de entrenamiento y prueba usando la función de import_df.py
    training_data, test_data = load_data()

    # Graficar algunos dígitos antes de entrenar la red
    plot_digits(training_data, num_per_class=5)

    # Crear la red neuronal con la arquitectura deseada
    net = Network(architecture)
    
    # Entrenar la red usando 30 épocas, con mini-lotes de tamaño 10 y una tasa de aprendizaje de 3.0
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
    
    # Devuelve la red entrenada
    return net

if __name__ == "__main__":
    # Config default
    config = {
        'architecture': [784, 30, 10],    # Número de neuronas por capa
        'epochs': 30,                      # Número de épocas
        'mini_batch_size': 10,             # Tamaño de los mini-lotes
        'eta': 3.0                          # Tasa de aprendizaje
    }
    main(config)