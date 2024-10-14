from Talleres.Taller2.Solucion.parte_practica.network import Network
import numpy as np
import random

class Network2(Network):
    def __init__(self, sizes):
        """Llamamos al constructor de la clase padre (Network) para inicializar
        los pesos, biases y las capas."""
        super().__init__(sizes)  # Llamamos al constructor de Network

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, regularization='L2', test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        Includes an option for L1 or L2 regularization."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n, regularization)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta, lmbda, n, regularization='L2'):
        """Update the network's weights and biases by applying gradient descent 
        using backpropagation to a single mini batch. Includes L1 and L2 regularization."""
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        # Calculamos los gradientes usando backpropagation
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            gradient_b = [gb + db for gb, db in zip(gradient_b, delta_b)]
            gradient_w = [gw + dw for gw, dw in zip(gradient_w, delta_w)]
        
        # Aplicación de la regularización (L2 por defecto o L1 si se especifica)
        if regularization == 'L2':
            self.weights = [
                (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * gw
                for w, gw in zip(self.weights, gradient_w)
            ]
        elif regularization == 'L1':
            self.weights = [
                w - (eta * (lmbda / n) * np.sign(w)) - (eta / len(mini_batch)) * gw
                for w, gw in zip(self.weights, gradient_w)
            ]
        
        # Actualización de biases (sin regularización)
        self.biases = [
            b - (eta / len(mini_batch)) * gb
            for b, gb in zip(self.biases, gradient_b)
        ]