# Reconocimiento de dígitos con Machine Learning

## Descripción
Este proyecto implementa un ejemplo introductorio de Machine Learning para reconocer dígitos escritos a mano utilizando el dataset `digits` de scikit-learn.

## Objetivo
Entrenar un clasificador capaz de identificar los dígitos del 0 al 9 y evaluar su rendimiento mediante Accuracy, Precision, Recall, F1 Score, matriz de confusión, ROC y AUC.

## Dataset
El dataset `digits` contiene 1797 muestras de imágenes de dígitos. Cada imagen está representada mediante 64 características correspondientes a una cuadrícula de 8 × 8 píxeles. Las etiquetas representan los dígitos del 0 al 9.

## Tecnologías
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Modelo
Se utilizó `LogisticRegression` con `max_iter=2000` y `solver="lbfgs"`. Se eligió porque es relativamente sencillo de explicar, funciona bien como clasificador multiclase y proporciona probabilidades mediante `predict_proba`, necesarias para construir ROC y calcular AUC.

## Proceso
Datos → Exploración → Visualización → Preparación → Entrenamiento → Predicción → Evaluación

## Métricas
- Accuracy: proporción de predicciones correctas.
- Precision: proporción de predicciones positivas que fueron correctas.
- Recall: proporción de casos reales recuperados correctamente.
- F1 Score: equilibrio entre Precision y Recall.
- ROC: relación entre TPR y FPR al variar el umbral.
- AUC: área bajo la curva ROC.

## Resultados reales
Con una división estratificada 80/20 y `random_state=42`:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9583 |
| Precision ponderada | 0.9590 |
| Recall ponderado | 0.9583 |
| F1 Score ponderado | 0.9584 |
| AUC macro OvR | 0.9989 |

El modelo obtuvo un rendimiento alto sobre el conjunto de prueba. La matriz de confusión permite identificar qué dígitos generaron los errores, mientras que la tabla F1 muestra el comportamiento por clase.

## Gráficas
- `graficas/curva_roc.png`
- `graficas/matriz_confusion.png`

## Estructura
```text
reconocimiento-digitos-ml/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── reconocimiento_digitos.ipynb
├── resultados/
│   ├── f1_score.csv
│   ├── roc.csv
│   └── metricas_modelo.csv
├── graficas/
│   ├── curva_roc.png
│   └── matriz_confusion.png
└── src/
    └── modelo.py
```

## Conclusiones
El modelo logra clasificar correctamente la gran mayoría de los ejemplos de prueba. El F1 ponderado es alto y el AUC macro OvR se encuentra muy próximo a 1, lo que indica una excelente capacidad de discriminación en este conjunto. Como mejoras futuras se podrían comparar otros algoritmos, ajustar hiperparámetros y probar datos externos escritos a mano.

## ¿Qué aprendí?
Una máquina puede reconocer un dígito transformando la imagen en características numéricas. En este dataset, cada imagen de 8 × 8 se representa mediante 64 valores de píxel. La etiqueta indica qué dígito representa la imagen. Durante el entrenamiento, el modelo aprende patrones que relacionan esas características con las etiquetas. Después, las predicciones sobre datos de prueba permiten comprobar si aprendió correctamente.

F1 Score ayuda a evaluar conjuntamente Precision y Recall. ROC muestra cómo cambia la tasa de verdaderos positivos frente a falsos positivos cuando varía el umbral. AUC resume esa capacidad de separación: cuanto más cerca de 1, mejor discriminación.

## Preguntas que podría hacer el profesor

1. **¿Qué dataset utilizaste?**  
   El dataset `digits` de scikit-learn.

2. **¿Cuántas muestras tiene?**  
   Tiene 1797 imágenes de dígitos.

3. **¿Qué representa cada píxel?**  
   Cada característica representa la intensidad de uno de los 64 píxeles de la imagen 8 × 8.

4. **¿Qué es Machine Learning?**  
   Es un enfoque en el que un modelo aprende patrones a partir de datos para realizar predicciones.

5. **¿Qué significa entrenar un modelo?**  
   Significa ajustar sus parámetros usando ejemplos conocidos para aprender la relación entre características y etiquetas.

6. **¿Por qué separaste entrenamiento y prueba?**  
   Para evaluar el modelo con datos que no utilizó durante el aprendizaje.

7. **¿Qué es Precision?**  
   Mide qué proporción de las predicciones positivas realizadas fueron correctas.

8. **¿Qué es Recall?**  
   Mide qué proporción de los casos reales de una clase fueron identificados correctamente.

9. **¿Qué es F1 Score?**  
   Es una medida que combina Precision y Recall mediante su media armónica.

10. **¿Qué es ROC?**  
    Es una curva que representa TPR frente a FPR para distintos umbrales.

11. **¿Qué es AUC?**  
    Es el área bajo la curva ROC y resume la capacidad de discriminación.

12. **¿Qué es FPR?**  
    Es la tasa de falsos positivos.

13. **¿Qué es TPR?**  
    Es la tasa de verdaderos positivos, equivalente al Recall en el contexto binario de cada clase.

14. **¿Por qué es multiclase?**  
    Porque el modelo debe distinguir diez clases diferentes: los dígitos del 0 al 9.

15. **¿Cómo sabes que el modelo funciona bien?**  
    En esta ejecución obtuvo Accuracy=0.9583, F1 ponderado=0.9584 y AUC macro OvR=0.9989. Estas métricas, junto con la matriz de confusión, muestran un rendimiento alto sobre los datos de prueba.

## GitHub

Desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Primer commit: reconocimiento de dígitos"
git branch -M main
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```

- `git init`: inicializa el repositorio local.
- `git add .`: prepara los archivos para el commit.
- `git commit`: crea una versión del proyecto.
- `git branch -M main`: establece `main` como rama principal.
- `git remote add origin`: conecta el proyecto con el repositorio remoto.
- `git push -u origin main`: sube el proyecto a GitHub.

**Importante:** reemplaza `URL_DEL_REPOSITORIO` por la URL real de tu repositorio. No se inventa ninguna URL.
