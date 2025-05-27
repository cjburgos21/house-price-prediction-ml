from sklearn.linear_model import LinearRegression


# Entrenamos el modelo con los datos escalados de entrenamiento
def entrenar_modelo(x_train, y_train):
    modelo = LinearRegression()
    modelo.fit(x_train, y_train)
    return modelo

