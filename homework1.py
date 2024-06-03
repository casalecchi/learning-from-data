import numpy as np
import matplotlib.pyplot as plt
import random


class Target:
    def __init__(self):
        a, b = self.get_coefficients()
        self.a = a
        self.b = b

    def get_coefficients(self):
        """Criação da função alvo. Retorna uma função para usar com os pontos e seus respectivos
        coeficientes, a e b"""
        # Gera pontos aleatórios
        x1, y1 = np.random.uniform(-1, 1, 2)
        x2, y2 = np.random.uniform(-1, 1, 2)
        # Calcula coeficientes da reta
        a = (y2 - y1) / (x2 - x1)
        b = y2 - a * x2
        return a, b
    
    # Definição da função
    def fit(self, X):
        """Dado um dataset X, retorna o seu y correspondente com o sinal em relação a função 
        alvo"""
        return np.sign(X[:, 1] - (self.a * X[:, 0] + self.b))

    def plot(self, X, y):
        "Visualização da função alvo e do Dataset criado"
        # Plotando os pontos do dataset e sua classificação
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
        
        # Plotando a função target
        target_x = np.linspace(-1, 1, 100)
        target_y = self.a * target_x + self.b
        plt.plot(target_x, target_y)
        
        # Configuração do gráfico
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Target Function and Dataset')
        plt.grid(True)
        plt.show()

def get_dataset(N=100):
    "Gera pontos aleatórios no espaço [-1, 1] x [-1, 1] com N pontos"
    X = np.random.uniform(-1, 1, (N, 2))
    return X

# Criação da função, dataset e visualização
# target = Target()
# X = get_dataset()
# y = target.fit(X)
# target.plot(X, y)

class PLA:
    def __init__(self, X):
        # vetor de pesos inicializado com 0
        self.w = np.zeros(3)
        # X modificado para ter nova coluna x0 com valor igual a 1
        self.X = np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, y, w=None):
        # reseta os pesos
        w = np.zeros(3) if not hasattr(w, 'shape') else w
        num_iters = 0
        
        while True:
            y_pred = np.sign(np.dot(self.X, w))
            misclassified = np.where(y_pred != y)[0]
            if len(misclassified) == 0:
                self.w = w
                return num_iters
            point = np.random.choice(misclassified)
            w += y[point] * self.X[point]
            num_iters += 1

    def divergence(self, target: Target):
        X_test = get_dataset(self.X.shape[0])
        y_test = target.fit(X_test)
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        y_pred = np.sign(np.dot(X_test, self.w))
        error = np.mean(y_test != y_pred)
        return error
    
def pla_plot(target: Target, X):
    "Visualização da função alvo e do Dataset criado"
    # y - classificação correta de acordo com função alvo
    y = target.fit(X)
    # Plotando os pontos do dataset e sua classificação
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
    
    # criamos e treinamos o perceptron
    pla = PLA(X)
    pla.fit(y)

    # Plotando a função f
    xs = np.linspace(-1, 1, 100)
    ys = target.a * xs + target.b
    plt.plot(xs, ys, label='$f$')

    # Plotando a função g
    xs = np.linspace(-1, 1, 100)
    ys = (-pla.w[1] * xs - pla.w[0]) / pla.w[2]
    plt.plot(xs, ys, label='$g$')
    
    # Configuração do gráfico
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$f$ and $g$ functions')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def perceptron_run(N, runs=1000):
    iterations = 0
    total_error = 0
    for _ in range(runs):
        target = Target()
        X = get_dataset(N)
        y = target.fit(X)
        pla = PLA(X)
        iterations += pla.fit(y)
        total_error += pla.divergence(target)
    mean_iterations = iterations / runs
    mean_divergence = total_error / runs
    print(f"N={N} - Iterações médias: {mean_iterations}")
    print(f"N={N} - Divergência média entre f e g: {mean_divergence}")

# primeiro e segundo experimentos
# perceptron_run(10)
# perceptron_run(100)
# Plotando funções f e g com os pontos
# target = Target()
# X = get_dataset(100)
# pla_plot(target, X)

# ---------------------- LINEAR REGRESSION --------------------------

class LinearRegression:
    def __init__(self, X):
        # inicializa pesos em 0
        self.w = np.zeros(3)
        # adiciona coluna x0 com 1
        self.X = np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, y):
        # atualiza os pesos com a multiplicação da pseudo-inversa com y
        self.w = np.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(y)
    
    def error(self, y, X=None):
        # identifica se deve adicionar a coluna em X ou se X já existe na instância
        X = self.X if not hasattr(X, 'shape') else np.c_[np.ones(X.shape[0]), X]
        # faz a multiplicação de X com os pesos para prever o resultado
        y_pred = np.sign(X.dot(self.w))
        # retorna média de resultados errados
        return np.mean(y_pred != y)

def lin_reg_run(N, runs=1000):
    ein = 0
    eout = 0
    for _ in range(runs):
        # cria função target, gera dados e respostas corretas
        target = Target()
        X = get_dataset(N)
        y = target.fit(X)

        # treina a regressão linear com os dados seguindo a função target
        LR = LinearRegression(X)
        LR.fit(y)
        # calcula erro de dentro da amostra
        ein += LR.error(y)

        # gero novos dados e novas respostas seguindo a função target
        new_X = get_dataset(N)
        new_y = target.fit(new_X)
        # calcula o erro dos novos dados seguindo o cálculo da Regressão Linear
        eout += LR.error(new_y, new_X)
    
    mean_ein = ein / runs
    mean_eout = eout / runs
    print(f"Erro médio dentro da amostra: Ein = {mean_ein}")
    print(f"Erro médio fora da amostra: Eout = {mean_eout}")

N = 100
lin_reg_run(N)

def LR_PLA_run(N, runs=1000):
    iterations = 0
    for _ in range(runs):
        # gera função target e pega os pesos da LR
        target = Target()
        X = get_dataset(N)
        y = target.fit(X)
        LR = LinearRegression(X)
        LR.fit(y)
        w = LR.w

        # faz o fit com o peso inicial igual ao da LR e calcula as iterações
        pla = PLA(X)
        iterations += pla.fit(y, w)
    
    mean_iterations = iterations / runs
    print(f"N={N} - Iterações médias com pesos iniciais da LR: {mean_iterations}")

N = 10
LR_PLA_run(N)