import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
# 1. Geração de Dados de Exemplo (os mesmos do scikit-learn para comparação)
np.random.seed(0)
X_sm = 2 * np.random.rand(100, 1)
y_sm = 4 + 3 * X_sm + np.random.randn(100, 1)
# Adicionar uma constante à variável independente para o StatsModels
# StatsModels não adiciona automaticamente o termo de intercepto, precisamos
fazer isso manualmente
X_sm = sm.add_constant(X_sm)
# 2. Criação e Treinamento do Modelo
# O modelo OLS (Ordinary Least Squares) é usado para Regressão Linear
model_sm = sm.OLS(y_sm, X_sm)
results_sm = model_sm.fit()
# 3. Exibição do Resumo do Modelo
print(results_sm.summary())
# 4. Extração de Coeficientes e Previsões
# O primeiro coeficiente é o intercepto (constante), o segundo é o coeficiente
de X
intercept_sm = results_sm.params[0]
coef_sm = results_sm.params[1]
print(f"\nIntercepto (β₀) do StatsModels: {intercept_sm:.2f}")
print(f"Coeficiente Angular (β₁) do StatsModels: {coef_sm:.2f}")
y_pred_sm = results_sm.predict(X_sm)
# 5. Visualização da Linha de Regressão
plt.figure(figsize=(8, 6))
plt.scatter(X_sm[:, 1], y_sm, alpha=0.7, label='Dados Reais') # X_sm[:, 1] para
pegar a coluna original de X
plt.plot(X_sm[:, 1], y_pred_sm, color='red', linewidth=2, label='Linha de
Regressão Prevista (StatsModels)')
plt.title('Regressão Linear com StatsModels')
plt.xlabel('Variável Independente (X)')
plt.ylabel('Variável Dependente (y)')
plt.legend()
plt.grid(True)
plt.show()