from sklearn.metrics import mean_squared_error

#Evaluando el modelo seleccionado

def evaluar_modelo(modelo, x_test, y_test):
    y_pred = modelo.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse