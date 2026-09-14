# Importamos NumPy para trabajar con arreglos numéricos y las diez clases del problema.
import numpy as np  # NumPy permite manipular datos numéricos de forma eficiente.
# Importamos el dataset de dígitos que proporciona scikit-learn para ejemplos educativos de clasificación.
from sklearn.datasets import load_digits  # load_digits carga imágenes pequeñas de dígitos escritos a mano.
# Importamos la función que separa los datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split  # Permite crear una división reproducible del dataset.
# Importamos la regresión logística para construir el clasificador.
from sklearn.linear_model import LogisticRegression  # Es un modelo de clasificación sencillo y explicable.
# Importamos las métricas necesarias para evaluar el modelo.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Calculamos métricas globales.
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score  # Calculamos evaluación multiclase.
# Importamos la función que transforma las clases en indicadores binarios para ROC One-vs-Rest.
from sklearn.preprocessing import label_binarize  # Convierte las diez clases en diez problemas binarios.

# Cargamos el dataset real de dígitos.
digits = load_digits()  # Obtenemos las imágenes, características y etiquetas del conjunto educativo.
# Extraemos las características de los píxeles y las etiquetas de los dígitos.
X, y = digits.data, digits.target  # X contiene 64 píxeles por muestra y y contiene el dígito correcto.
# Separamos 80 % para entrenamiento y 20 % para prueba y conservamos la proporción de las clases.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)  # random_state hace reproducible la división.
# Creamos el modelo de regresión logística.
model = LogisticRegression(max_iter=2000, solver="lbfgs")  # max_iter evita que el optimizador termine demasiado pronto.
# Entrenamos el modelo solamente con los datos de entrenamiento.
model.fit(X_train, y_train)  # El modelo aprende la relación entre los píxeles y el dígito.
# Generamos las clases predichas para el conjunto de prueba.
y_pred = model.predict(X_test)  # Estas predicciones se comparan con las etiquetas reales.
# Generamos probabilidades para cada clase, necesarias para ROC y AUC.
y_score = model.predict_proba(X_test)  # Cada fila contiene la probabilidad estimada para los dígitos 0 a 9.

# Calculamos accuracy para conocer la proporción total de predicciones correctas.
accuracy = accuracy_score(y_test, y_pred)  # Accuracy = aciertos divididos entre el total de ejemplos.
# Calculamos precision ponderada para considerar el soporte de cada clase.
precision = precision_score(y_test, y_pred, average="weighted")  # Precision indica qué tan fiables son las predicciones positivas.
# Calculamos recall ponderado para medir cuánto recuperamos de cada clase real.
recall = recall_score(y_test, y_pred, average="weighted")  # Recall indica qué proporción de ejemplos reales identificamos correctamente.
# Calculamos F1 ponderado como equilibrio entre precision y recall.
f1 = f1_score(y_test, y_pred, average="weighted")  # F1 combina precision y recall mediante su media armónica.
# Convertimos las etiquetas del conjunto de prueba a formato binario para ROC multiclase.
y_test_bin = label_binarize(y_test, classes=np.arange(10))  # Cada dígito se convierte en una columna binaria.
# Calculamos AUC macro One-vs-Rest dando el mismo peso a cada clase.
auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")  # AUC mide la capacidad de separar clases.
# Mostramos el resumen de métricas en pantalla.
print(f"Accuracy: {accuracy:.4f}")  # Mostramos accuracy con cuatro decimales.
print(f"Precision ponderada: {precision:.4f}")  # Mostramos precision ponderada.
print(f"Recall ponderado: {recall:.4f}")  # Mostramos recall ponderado.
print(f"F1 Score ponderado: {f1:.4f}")  # Mostramos F1 ponderado.
print(f"AUC macro OvR: {auc_macro:.4f}")  # Mostramos AUC macro One-vs-Rest.
