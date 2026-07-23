# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}


# flake8: noqa: E501

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# -------------------------------------------------------------------------
# Rutas de los archivos
# -------------------------------------------------------------------------

TRAIN_FILENAME = "files/input/train_data.csv.zip"
TEST_FILENAME = "files/input/test_data.csv.zip"

MODEL_FILENAME = "files/models/model.pkl.gz"
METRICS_FILENAME = "files/output/metrics.json"


# -------------------------------------------------------------------------
# Paso 1. Limpieza de los datos
# -------------------------------------------------------------------------


def clean_data(data):
    """
    Limpia el conjunto de datos de tarjetas de crédito.
    """

    data = data.copy()

    # Renombrar la variable objetivo
    data = data.rename(columns={"default payment next month": "default"})

    # Eliminar el identificador
    data = data.drop(columns=["ID"])

    # Eliminar registros con valores faltantes
    data = data.dropna()

    # EDUCATION = 0 y MARRIAGE = 0 representan información no disponible
    data = data[(data["EDUCATION"] != 0) & (data["MARRIAGE"] != 0)].copy()

    # Agrupar niveles de educación mayores que 4 en "others"
    data.loc[data["EDUCATION"] > 4, "EDUCATION"] = 4

    return data


# -------------------------------------------------------------------------
# Paso 2. Separación entre variables explicativas y variable objetivo
# -------------------------------------------------------------------------


def split_features_target(data):
    """
    Separa las variables explicativas y la variable objetivo.
    """

    x = data.drop(columns=["default"])
    y = data["default"]

    return x, y


# -------------------------------------------------------------------------
# Pasos 3 y 4. Pipeline y optimización de hiperparámetros
# -------------------------------------------------------------------------


def build_model():
    """
    Construye el pipeline y el GridSearchCV.
    """

    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    # Las variables categóricas se transforman mediante OneHotEncoder.
    # Las demás variables se estandarizan con StandardScaler.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder=StandardScaler(),
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("select_k_best", SelectKBest(k=12)),
            ("classifier", SVC(gamma=0.1)),
        ]
    )

    parameters = {
        "pca__n_components": [20, 21],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return model


# -------------------------------------------------------------------------
# Pasos 6 y 7. Métricas y matrices de confusión
# -------------------------------------------------------------------------


def create_metrics_record(dataset, y_true, y_pred):
    """
    Construye un registro con las métricas del modelo.
    """

    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


def create_confusion_matrix_record(dataset, y_true, y_pred):
    """
    Construye un registro con la matriz de confusión.
    """

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred
    ).ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(true_negative),
            "predicted_1": int(false_positive),
        },
        "true_1": {
            "predicted_0": int(false_negative),
            "predicted_1": int(true_positive),
        },
    }


# -------------------------------------------------------------------------
# Ejecución principal
# -------------------------------------------------------------------------


def main():
    """
    Ejecuta todos los pasos solicitados por el laboratorio.
    """

    # Cargar los datos comprimidos directamente
    train_data = pd.read_csv(TRAIN_FILENAME)
    test_data = pd.read_csv(TEST_FILENAME)

    # Limpiar los datos
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    # Separar variables explicativas y objetivo
    x_train, y_train = split_features_target(train_data)
    x_test, y_test = split_features_target(test_data)

    # Crear y entrenar el modelo
    model = build_model()
    model.fit(x_train, y_train)

    # Crear las carpetas de salida cuando no existan
    os.makedirs(
        os.path.dirname(MODEL_FILENAME),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(METRICS_FILENAME),
        exist_ok=True,
    )

    # Paso 5. Guardar el modelo comprimido con gzip
    with gzip.open(MODEL_FILENAME, "wb") as file:
        pickle.dump(model, file)

    # Realizar predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # El orden es importante para el autograder
    records = [
        create_metrics_record(
            "train",
            y_train,
            y_train_pred,
        ),
        create_metrics_record(
            "test",
            y_test,
            y_test_pred,
        ),
        create_confusion_matrix_record(
            "train",
            y_train,
            y_train_pred,
        ),
        create_confusion_matrix_record(
            "test",
            y_test,
            y_test_pred,
        ),
    ]

    # Guardar cada diccionario en una línea independiente
    with open(
        METRICS_FILENAME,
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()