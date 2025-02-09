# Desafio_Ciencia_Dados

Para executar este projeto, siga os seguintes passos:

Clone o repositório:

git clone https://github.com/AmandamMoreira/Desafio_Ciencia_Dados.git

Acesse o diretório do projeto:

cd Desafio_Ciencia_Dados

# Crie um ambiente virtual (opcional, mas recomendado):
python -m venv venv
source venv/bin/activate  # Para Linux/Mac
venv\Scripts\activate  # Para Windows

# Instale as dependências:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from collections import Counter
import joblib
Para instala-las pode executar no prompt: pip install (nome do modulo)

# Como Executar
Executar o Jupyter Notebook

# Para abrir os notebooks de análise exploratória e modelagem, execute:
jupyter notebook
Em seguida, abra os arquivos dentro da pasta notebooks/.

# Carregar e Usar o Modelo Treinado

# Para carregar o modelo salvo e fazer previsões, utilize o seguinte código:
import joblib

# Carregar o modelo salvo
modelo_carregado = joblib.load("modelo_xgboost.pkl")

print("Modelo carregado com sucesso!")

import pandas as pd

# Criar um DataFrame com TODAS as colunas que o modelo espera
novo_exemplo = pd.DataFrame()

# Fazer a previsão
previsao = modelo_carregado.predict(novo_exemplo)
print(f"Preço previsto: {previsao[0]:.2f}")
