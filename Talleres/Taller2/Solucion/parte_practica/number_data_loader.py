import tensorflow as tf
import numpy as np

def vectorized_result(j):
    """Retorna un vector columna de 10 dimensiones con 1.0 en la posición j y 0.0 en las demás."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data():
    """Cargar los datos de MNIST y retornarlos en el formato adecuado."""
    # Cargar los datos de MNIST
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reestructurar las imágenes a vectores de tamaño 784 (28x28) y normalizar a [0, 1]
    train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

    # Convertir las etiquetas en formato one-hot para training_data
    train_labels_one_hot = [vectorized_result(y) for y in train_labels]

    # Empaquetar los datos en listas de tuplas (imagen, etiqueta) para compatibilidad con el código de la red
    training_data = list(zip([x.reshape(784, 1) for x in train_images], train_labels_one_hot))

    # Test_data con etiquetas como enteros
    test_data = list(zip([x.reshape(784, 1) for x in test_images], test_labels))

    return training_data, test_data