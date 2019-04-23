# Importar librerías

import keras 
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd

# Importar la información

data = pd.read_csv('concrete_data.csv')

# Declaración de variables imporantes

data_columnas = data.columns
labels = data[data_columnas[data_columnas != 'Strength']]
target = data['Strength']
n_cols = labels.shape[1]

# Crear el modelo

modelo = Sequential()

# Agregar capas
# [modelo].add(Dense([50], activation = [metodo], input_shape = [int]))
modelo.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
modelo.add(Dense(50, activation = 'relu'))
modelo.add(Dense(1))

# Compilar 
# # [modelo].compile(optimizer = 'funcion', loss = 'mean_squared_error')
modelo.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Entrenar
modelo.fit(labels, target, validation_split = 0.3, epochs = 100, verbose = 2)