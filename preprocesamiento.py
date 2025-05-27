import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

#Cargando el dataset a utilizar
def cargar_datos():
    df = pd.read_csv(config.DATASET_PATH)
    return df


#Limpiar el dataset (eliminando valores nulos, atípicos y transformando datos categóricos)
def limpieza(df):
    df = df.dropna() #Eliminamos las filas con los valores nulos
    df = df[df["median_house_values"] < 500000]
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True) #Eliminamos la siguiente columna categorica
    return df

#Dividir el dataset en variables independientes y dependientes
def dividir(df):
    x = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    return train_test_split(x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

#Normalizacion de los datos
def normalizar(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train) #Ajusta la escala(calculo de media y desviación estándar) y transforma el set de entrenamiento
    x_test_scaled = scaler.transform(x_test) #Hace lo mismo pero para el set de prueba
    return x_train_scaled, x_test_scaled