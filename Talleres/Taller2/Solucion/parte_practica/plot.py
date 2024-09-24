import numpy as np
import matplotlib.pyplot as plt

def plot_digits(data, num_per_class=5):
    """
    Grafica una cantidad específica de dígitos por clase.

    Parámetros:
    - data (list of tuples): Lista de tuplas (x, y) donde x es la imagen (en forma de array) y y es la etiqueta en formato one-hot.
    - num_per_class (int): Número de ejemplos de cada dígito que se quiere graficar.

    Esta función selecciona ejemplos de las clases de dígitos del 0 al 9,
    crea una cuadrícula de imágenes y muestra dichos ejemplos en una figura.
    """
    examples = {}  # Diccionario para almacenar ejemplos por clase
    for x, y in data:
        label = np.argmax(y)  # Obtener la etiqueta de la salida one-hot
        if label not in examples:
            examples[label] = []  # Si no existe la clave de la clase, la crea
        if len(examples[label]) < num_per_class:
            examples[label].append(x)  # Agregar la imagen a la lista de ejemplos de la clase
        # Verificar si ya tenemos suficientes ejemplos para todas las clases
        if all(len(v) >= num_per_class for v in examples.values()):
            break

    # Ordenar las clases y crear la figura para graficar
    classes = sorted(examples.keys())  # Clases ordenadas
    num_classes = len(classes)  # Número total de clases
    plt.figure(figsize=(num_per_class * 2, num_classes * 2))  # Tamaño de la figura

    # Graficar cada ejemplo
    for idx, cls in enumerate(classes):
        for i in range(num_per_class):
            plt_idx = idx * num_per_class + i + 1  # Índice en la cuadrícula
            plt.subplot(num_classes, num_per_class, plt_idx)  # Crear subgráfico
            img = examples[cls][i].reshape(28, 28)  # Redimensionar la imagen a 28x28
            plt.imshow(img, cmap='gray')  # Mostrar la imagen en escala de grises
            plt.axis('off')  # Ocultar los ejes
            if i == 0:
                plt.ylabel(f'Dígito {cls}', fontsize=12)  # Etiquetar las filas
    plt.suptitle('Ejemplos de Dígitos de Entrenamiento', fontsize=16)  # Título de la figura
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar el espaciado
    plt.show()  # Mostrar la figura

def plot_predicted_digits(network, data, num_per_class=5, title='Ejemplos de Dígitos Predichos por la Red'):
    """
    Grafica una cantidad específica de dígitos predichos por la red por clase.

    Parámetros:
    - network (Network): Instancia de la red neuronal entrenada.
    - data (list of tuples): Lista de tuplas (x, y) donde x es la imagen (en forma de array) y y es la etiqueta real.
    - num_per_class (int): Número de ejemplos de cada dígito que se quiere graficar.

    Esta función selecciona ejemplos de las clases de dígitos del 0 al 9,
    crea una cuadrícula de imágenes y muestra dichos ejemplos en una figura.
    """
    examples = {}  # Diccionario para almacenar ejemplos por clase
    for x, y in data:
        predicted_label = np.argmax(network.feedforward(x))  # Obtener la etiqueta predicha por la red
        if predicted_label not in examples:
            examples[predicted_label] = []  # Si no existe la clave de la clase, la crea
        if len(examples[predicted_label]) < num_per_class:
            examples[predicted_label].append(x)  # Agregar la imagen a la lista de ejemplos de la clase
        # Verificar si ya tenemos suficientes ejemplos para todas las clases
        if all(len(v) >= num_per_class for v in examples.values()):
            break

    # Ordenar las clases y crear la figura para graficar
    classes = sorted(examples.keys())  # Clases ordenadas
    num_classes = len(classes)  # Número total de clases
    plt.figure(figsize=(num_per_class * 2, num_classes * 2))  # Tamaño de la figura

    # Graficar cada ejemplo
    for idx, cls in enumerate(classes):
        for i in range(num_per_class):
            plt_idx = idx * num_per_class + i + 1  # Índice en la cuadrícula
            plt.subplot(num_classes, num_per_class, plt_idx)  # Crear subgráfico
            img = examples[cls][i].reshape(28, 28)  # Redimensionar la imagen a 28x28
            plt.imshow(img, cmap='gray')  # Mostrar la imagen en escala de grises
            plt.axis('off')  # Ocultar los ejes
            if i == 0:
                plt.ylabel(f'Predicción {cls}', fontsize=12)  # Etiquetar las filas
    plt.suptitle(title, fontsize=16)  # Título de la figura
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar el espaciado
    plt.show()  # Mostrar la figura