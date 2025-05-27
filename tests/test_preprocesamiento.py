import pytest
import pandas as pd
from preprocesamiento import cargar_datos, limpieza

def test_cargar_datos():
    df = cargar_datos()
    assert isinstance(df, pd.DataFrame)
    assert 'SalePrice' in df.columns

def test_limpiar_datos():
    df = cargar_datos()
    df_limpio = limpieza(df)
    assert df_limpio.isnull().sum().sum() == 0 #Verificar que no existan valores nulos

