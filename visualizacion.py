import matplotlib.pyplot as plt

def graficar_resultados(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Valores Real")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.grid(True)
    plt.show()