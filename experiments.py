import numpy as np
import matplotlib.pyplot as plt
import random


class Target:
    """Classe para criação e manipulação de um a função target"""
    def __init__(self):
        a, b = self.get_coefficients()
        self.a = a
        self.b = b

    def get_coefficients(self):
        # Gera pontos aleatórios
        x1, y1 = np.random.uniform(-1, 1, 2)
        x2, y2 = np.random.uniform(-1, 1, 2)
        # Calcula coeficientes da reta
        a = (y2 - y1) / (x2 - x1)
        b = y2 - a * x2
        return a, b
    
    # Definição da função
    def fit(self, X):
        # Retorna o y dado um X, de uma função target
        return np.sign(X[:, 1] - (self.a * X[:, 0] + self.b))

    def plot(self, X, y):
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
    """Classe para criação e manipulação de um Perceptron Learning Algorithm"""
    def __init__(self, X):
        # vetor de pesos inicializado com 0
        self.w = np.zeros(3)
        # X modificado para ter nova coluna x0 com valor igual a 1
        self.X = np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, y, w=np.zeros(3)):
        # peso inicial, se não for passado, igual a zero
        num_iters = 0
        
        while True:
            # cálculo de y com pesos atuais
            y_pred = np.sign(np.dot(self.X, w))
            
            # separar y classificados erroneamente, verificar se convergiu e escolher um aleatoriamente
            misclassified = np.where(y_pred != y)[0]
            if len(misclassified) == 0:
                self.w = w
                return num_iters
            point = np.random.choice(misclassified)
            
            # ajustar peso para ponto escolhido
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
        # verifica dado uma função target, o quanto a hipótese g difere
        # gera dados de teste
        X_test = get_dataset(self.X.shape[0])
        y_test = target.fit(X_test)
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        y_pred = np.sign(np.dot(X_test, self.w))
        # compara as classificações da função target com as geradas pelo PLA
        error = np.mean(y_test != y_pred)
        return error
    
def pla_plot(target: Target, X):
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
    plt.plot(xs, ys, label='$f$: target function')

    # Plotando a função g
    xs = np.linspace(-1, 1, 100)
    ys = (-pla.w[1] * xs - pla.w[0]) / pla.w[2]
    plt.plot(xs, ys, label='$g$: PLA hypotesys')
    
    # Configuração do gráfico
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$f$ and $g$ functions')
    plt.grid(True)
    plt.legend()
    plt.show()
    
# Experimento da questão
def perceptron_run(N, runs=1000):
    iterations = 0
    total_error = 0
    for _ in range(runs):
        # cria função target e dados
        target = Target()
        X = get_dataset(N)
        y = target.fit(X)
        
        # cria PLA, treino e divergência da função target
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

# -------------------------------- LINEAR REGRESSION -------------------------------------

class LinearRegression:
    """Classe para criação e manipulação da Regressão Linear"""
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

# Experimento da questão
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

# Experimento da questão
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

# Definindo a classe NonLinear, pois a função de gerar um dataset com noise vai utilizá-la também
class NonLinear:
    """Classe para criação e manipulação de funções target não lineares"""
    def __init__(self):
        pass
    
    def fit(self, X):
        # função passada no enunciado
        return np.sign(X[:, 0]**2 + X[:, 1]**2 -0.6)
    
    def transform(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        # pega os x1 e x2 e transforma
        # não adicionei a coluna x0 com valor 1, pois na regressão linear ela será adicionada
        X_trans = np.c_[x1, x2, x1 * x2, x1**2, x2**2]
        return X_trans

def get_noisy_dataset(N, target: Target | NonLinear, noise=0.1):
    # cria dataset e pega o seu valor correto
    X = get_dataset(N)
    y = target.fit(X)

    # escolhe pontos aleatórios e inverte sua classificação -> retorna X e y
    points = int(N * noise)
    noisy_indices = np.random.choice(N, points, replace=False)
    y[noisy_indices] = -y[noisy_indices]

    return X, y

def plot_pocket_PLA(X, y, w, target:Target, title):
    # Geração de gráfico do pocket PLA
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

# Experimento da questão
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

# for key, value in topics.items():
#     print(key)
#     pocket_PLA_run(*value)
    

# -------------------------------------- NON LINEAR REGRESSION -------------------------------------------

# Classe NonLinear já definida

# Experimento da questão
def non_linear_run(N, runs=1000):
    ein = 0
    for _ in range(runs):
        # geração de dados não lineares com ruído
        nonlinear = NonLinear()
        X_noisy, y_noisy = get_noisy_dataset(N, nonlinear)
        
        # criação e treinamento do modelo de regressão linear
        LR = LinearRegression(X_noisy)
        LR.fit(y_noisy)
        w = LR.w

        # cálculo do erro
        ein += get_error(X_noisy, y_noisy, w)
    
    mean_ein = ein / runs
    print(f'Média do Erro dentro da Amostra: Ein = {mean_ein}')

#N = 1000
# non_linear_run(N)

# Experimento da questão
def transform_run(N, runs=1000):
    ws = []
    for _ in range(runs):
        # geração de dados não lineares com ruído
        nonlinear = NonLinear()
        X_noisy, y_noisy = get_noisy_dataset(N, nonlinear)
        # transformação nos dados
        X_trans = nonlinear.transform(X_noisy)

        # criação e treinamento do modelo de regressão linear
        LR = LinearRegression(X_trans)
        LR.fit(y_noisy)
        w = LR.w
        ws.append(w)
    
    mean_ws = np.mean(ws, axis=0)
    print(f'Pesos médios: {mean_ws}')
    return mean_ws

# ws = transform_run(N)

# options = {
#     "a": np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5]),
#     "b": np.array([-1, -0.05, 0.08, 0.13, 1.5, 15]),
#     "c": np.array([-1, -0.05, 0.08, 0.13, 15, 1.5]),
#     "d": np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05]),
#     "e": np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15]),
# }

# closest_hypothesis = min(options, key=lambda k: np.linalg.norm(ws - options[k]))
# print(f"Hipótese escolhida: {closest_hypothesis}")

# Experimento da questão
def eout_nonlinear_run(N, runs=1000):
    eout = 0
    for _ in range(runs):
        # geração de dados não lineares com ruído
        nonlinear = NonLinear()
        X_noisy, y_noisy = get_noisy_dataset(N, nonlinear)
        # transformação nos dados
        X_trans = nonlinear.transform(X_noisy)

        # criação e treinamento do modelo de regressão linear
        LR = LinearRegression(X_trans)
        LR.fit(y_noisy)
        w = LR.w

        # criação e transformação dos dados de teste
        X_test = get_dataset(N)
        y_test = nonlinear.fit(X_test)
        X_test = nonlinear.transform(X_test)
        
        # cálculo do erro 
        eout += get_error(X_test, y_test, w)

    mean_eout = eout / runs
    print(f'Média do Erro fora da Amostra: Eout = {mean_eout}')

# eout_nonlinear_run(N)
