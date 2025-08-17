import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Geração de Dados de Exemplo
np.random.seed(0) # Para reprodutibilidade
X = 2 * np.random.rand(100, 1) # 100 pontos de dados entre 0 e 2
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + ruído

# 2. Visualização dos Dados
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.7)
plt.title('Dados Gerados para Regressão Linear')
plt.xlabel('Variável Independente (X)')
plt.ylabel('Variável Dependente (y)')
plt.grid(True)
plt.show()

# 3. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
print(f"Tamanho do conjunto de teste: {len(X_test)} amostras")

# 4. Criação e Treinamento do Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = model.predict(X_test)
# Coeficientes do modelo
print(f"\nCoeficiente Angular (β₁): {model.coef_[0][0]:.2f}")
print(f"Intercepto (β₀): {model.intercept_[0]:.2f}")
# Métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# 6. Visualização da Linha de Regressão
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Dados Reais de Teste')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linha de RegressãoPrevista')
plt.title('Regressão Linear: Dados de Teste e Linha de Regressão')
plt.xlabel('Variável Independente (X)')
plt.ylabel('Variável Dependente (y)')
plt.legend()
plt.grid(True)
plt.show()

# 7. Fazendo Previsões com o Modelo Treinado
novos_X = np.array([[1.5], [0.8], [2.2]])
previsoes = model.predict(novos_X)
print("\nPrevisões para novos dados:")
for i, x_val in enumerate(novos_X):
    print(f"X = {x_val[0]:.2f}, Y Previsto = {previsoes[i][0]:.2f}")