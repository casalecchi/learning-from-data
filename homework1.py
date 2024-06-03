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
    
    def fit(self, y, w=np.zeros(3)):
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
    
    def pocket_fit(self, X, y, max_iterations, w_init=np.zeros(3)):
        # adiciona coluna x0 igual a 1
        X = np.c_[np.ones(X.shape[0]), X]
        # guardar melhor peso
        best_w = w_init
        # guardar melhor erro - igual ao erro com peso inicial
        best_error = np.mean(np.sign(X.dot(w_init)) != y)

        w = w_init.copy()
        for _ in range(max_iterations):
            # resultados com peso atual
            y_pred = np.sign(X.dot(w))
            
            # pega um ponto classificado erroneamente
            misclassified = np.where(y_pred != y)[0]
            if len(misclassified) == 0:
                break
            
            # cálculo do peso para o ponto
            point = np.random.choice(misclassified)
            w += y[point] * X[point]
            
            # cálculo do erro atual
            current_error = np.mean(np.sign(X.dot(w)) != y)

            if current_error < best_error:
                # guarda erro e peso
                best_error = current_error
                best_w = w.copy()
        
        return best_w, best_error

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
    
def get_error(X, y, w):
        X = np.c_[np.ones(X.shape[0]), X]
        # faz a multiplicação de X com os pesos para prever o resultado
        y_pred = np.sign(X.dot(w))
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
        w = LR.w
        # calcula erro de dentro da amostra
        ein += get_error(X, y, w)

        # gero novos dados e novas respostas seguindo a função target
        new_X = get_dataset(N)
        new_y = target.fit(new_X)
        # calcula o erro dos novos dados seguindo o cálculo da Regressão Linear
        eout += get_error(new_X, new_y, w)
    
    mean_ein = ein / runs
    mean_eout = eout / runs
    print(f"Erro médio dentro da amostra: Ein = {mean_ein}")
    print(f"Erro médio fora da amostra: Eout = {mean_eout}")

# N = 100
# lin_reg_run(N)

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

# N = 10
# LR_PLA_run(N)

def get_noisy_dataset(N, target: Target, noise=0.1):
    # cria dataset e pega o seu valor correto
    X = get_dataset(N)
    y = target.fit(X)

    # escolhe pontos aleatórios e inverte sua classificação -> retorna X e y
    points = int(N * noise)
    noisy_indices = np.random.choice(N, points, replace=False)
    y[noisy_indices] = -y[noisy_indices]

    return X, y

def plot_pocket_PLA(X, y, w, target:Target, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.9)
    
    # Plotando função alvo
    xs = np.linspace(-1, 1, 100)
    ys = target.a * xs + target.b
    plt.plot(xs, ys, label='$f$: Target Function')
    # Plotando a função g
    xs = np.linspace(-1, 1, 100)
    ys = (-w[1] * xs - w[0]) / w[2]
    plt.plot(xs, ys, label='$g$: Pocket PLA')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

def pocket_PLA_run(N1, N2, max_iter, init_LR, runs=1000):
    ein = 0
    eout = 0
    for _ in range(runs):
        # cria função alvo e dataset com valores invertidos
        target = Target()
        X_train, y_train = get_noisy_dataset(N1, target)
        
        # cria pla para usar o pocket pla
        pla = PLA(X_train)
        
        # cria Linear Regression para pegar os pesos
        LR = LinearRegression(X_train)
        LR.fit(y_train)
        w_LR = LR.w
        
        # usa o peso ou não
        if init_LR:
            w_pocket, best_Ein = pla.pocket_fit(X_train, y_train, max_iter, w_init=w_LR)
        else:
            w_pocket, best_Ein = pla.pocket_fit(X_train, y_train, max_iter)

        # cria dataset correto e testa com a função treinada
        X_test = get_dataset(N2)
        y_test = target.fit(X_test)
        
        # adiciona o erro para a média final depois ser calculada
        eout += get_error(X_test, y_test, w_pocket)
        ein += best_Ein

    # Plotando gráfico com a função target e a função gerado pelo Pocket PLA
    plot_pocket_PLA(X_test, y_test, w_pocket, target, title=f'Test Dataset for Pocket PLA with {'Linear Regression' if init_LR else 'No'} Initialization\nand {max_iter} iterations')
    
    mean_ein = ein / runs
    mean_eout = eout / runs
    print(f"Média do Erro dentro da Amostra: Ein = {mean_ein}")
    print(f"Média do Erro fora da Amostra: Eout = {mean_eout}")
    
N1 = 100
N2 = 1000

topics = {
    "a) W0=0, i=10, N1=100, N2=1000": [N1, N2, 10, False],
    "b) W0=0, i=50, N1=100, N2=1000": [N1, N2, 50, False],
    "c) W0=LR, i=10, N1=100, N2=1000": [N1, N2, 10, True],
    "d) W0=LR, i=50, N1=100, N2=1000": [N1, N2, 50, True],
}

for key, value in topics.items():
    print(key)
    pocket_PLA_run(*value)
    