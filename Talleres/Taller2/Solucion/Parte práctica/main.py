from import_df import load_data
from network import Network

def main():
    # Cargar los datos de entrenamiento y prueba usando la función de import_df.py
    training_data, test_data = load_data()

    # Crear la red neuronal con la arquitectura desead
    net = Network([77, 30, 10])
    
    # Entrenar la red usando 30 épocas, con mini-lotes de tamaño 10 y una tasa de aprendizaje de 3.0
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

    # La evaluación se realiza automáticamente durante el entrenamiento
    # al final de cada época usando los datos de prueba (test_data)

if __name__ == "__main__":
    main()
