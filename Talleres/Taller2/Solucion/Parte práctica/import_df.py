import tensorflow as tf
import numpy as np

def load_data():
    """Cargar los datos de MNIST y retornarlos en el formato adecuado."""
    # Cargar los datos de MNIST
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reestructurar las imágenes a vectores de tamaño 784 (28*28) y normalizar a [0, 1]
    train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

    # Empaquetar los datos en listas de tuplas para compatibilidad con el código de la red
    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))

    return training_data, test_data