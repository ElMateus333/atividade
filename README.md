# atividade
# EXERCICIO 1
def perceptron_input(inputs, weights):

    """
    :param inputs: list input values
    :param weights: list of weights corresponding to the inputs
    :return: the result of weighted sum inputs plus the bias
    """
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return weighted_sum

def perceptron_output(inputs, weights, threshold):

    """
    :param inputs: list of input values
    :param weights: list of weights corresponding to the inputs
    :param threshold: Activation threshold
    :return: the output of the perceptron
    """
    weighted_sum = perceptron_input(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

x = [1, 1, 2]

w = [1, 1, 0]

threshold = 1.5

output = perceptron_output(x, w, threshold)
print(output)
# EXERCICIO 2
import numpy as np

class Perceptron:

    def __init__(self, n_inputs, learning_rate=0.01):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def activation_function(self, x):
        # Função de ativação simples: degrau (threshold 0)
        return 1 if x >= 0 else 0

    def predict(self, X):
        # Calcula f(X_n, W_n, b) = b + Σ W_i,j * X_i
        linear_output = self.bias + np.dot(self.weights, X)
        return self.activation_function(linear_output)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                error = y[idx] - prediction
                
                # Atualiza os pesos e o bias
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

if __name__ == "__main__":

    # Dados de entrada (X) e classes (y)
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 0, 0, 0])  # Saídas esperadas

    # Inicializa o Perceptron
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1)

    # Treina o Perceptron
    perceptron.train(X, y, epochs=10)

    # Testa o Perceptron
    for x in X:
        print(f"Entrada: {x}, Previsão: {perceptron.predict(x)}")
Explicação:
Classe Perceptron: Implementa o Perceptron com pesos aleatórios e um bias.
Função de ativação: Usa a função degrau (step function) simples, retornando 1 se o resultado da soma for maior ou igual a 0 e 0 caso contrário.
Treinamento: O Perceptron é treinado ajustando os pesos e o bias com base no erro entre a previsão e o valor real.
Predição: A função de predição usa a fórmula 

0
,
0
]
]
X=[[1,1],[1,0],[0,1],[0,0]], a previsão do Perceptron será calculada para cada amostra de dados.
