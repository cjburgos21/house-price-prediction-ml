import pytest
from modelo import entrenar_modelo
from preprocesamiento import dividir, cargar_datos

def test_entrenar_modelo():
    df = cargar_datos()
    x_train, x_test, y_train, y_test = dividir(df)
    modelo_entrenado = entrenar_modelo(x_train, y_train)
    assert modelo_entrenado is not None