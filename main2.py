import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle

# Carregando o dataset de exemplo
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Salvando o modelo em um arquivo pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Carregando o modelo a partir do arquivo pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Título da aplicação
st.title('Previsão com Regressão Logística')

# Input das variáveis
sepal_length = st.slider('Sepal Length', 4.3, 7.9, 5.1)
sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.5)
petal_length = st.slider('Petal Length', 1.0, 6.9, 1.4)
petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

# Realizando a previsão com o modelo
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

# Exibindo o resultado da previsão
species = iris.target_names[prediction][0]
st.write(f'A espécie prevista é: {species}')