import preprocesamiento
import modelo
import evaluacion
import visualizacion
import salida

def main():
    #Llamando a las funciones en el orden necesario
    df = preprocesamiento.cargar_datos()
    df = preprocesamiento.limpieza(df)

    #Dividiendo datos, entrenamiento y prueba
    x_train, x_test, y_train, y_test = preprocesamiento.dividir(df)

    #Normalizacion
    x_train_scaled, x_test_scaled = preprocesamiento.normalizar(x_train, x_test)

    #Entrenando el modelo
    modelo_entrenado = modelo.entrenar_modelo(x_train_scaled, y_train)

    #Evaluar el modelo
    mse = evaluacion.evaluar_modelo(modelo_entrenado, x_test_scaled, y_test)

    #Mostrar resultados
    salida.mostrar_resultados(mse)

    #Visualizacion de resultados
    y_pred = modelo_entrenado.predict(x_test_scaled)
    visualizacion.graficar_resultados(y_test, y_pred)

if __name__ == "__main__":
    main()