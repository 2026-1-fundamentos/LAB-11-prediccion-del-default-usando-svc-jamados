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
#
# flake8: noqa: E501
import gzip
import json
import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==============================================================================
# Paso 1. Limpieza de los datasets
# ==============================================================================


def clean_dataset(df):
    df_clean = df.copy()

    # Renombrar columna objetivo
    if "default payment next month" in df_clean.columns:
        df_clean = df_clean.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    if "ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["ID"])

    # Eliminar registros con valores nulos o información no disponible (0 = N/A)
    df_clean = df_clean.dropna()
    if "EDUCATION" in df_clean.columns:
        df_clean = df_clean[df_clean["EDUCATION"] != 0]
    if "MARRIAGE" in df_clean.columns:
        df_clean = df_clean[df_clean["MARRIAGE"] != 0]

    # Para EDUCATION, valores > 4 se agrupan en "others" (4)
    if "EDUCATION" in df_clean.columns:
        df_clean["EDUCATION"] = df_clean["EDUCATION"].apply(
            lambda x: 4 if x > 4 else x
        )

    return df_clean


# Carga de datos
train_df = pd.read_csv("files/input/train_data.csv.zip")
test_df = pd.read_csv("files/input/test_data.csv.zip")

# Aplicar limpieza
train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)


# ==============================================================================
# Paso 2. División en X e y
# ==============================================================================
x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]
x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]


# ==============================================================================
# Paso 3. Construcción del Pipeline
# ==============================================================================
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

# Preprocesador para variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

# Pipeline con el orden estricto solicitado en el test
pipeline = Pipeline(
    steps=[
        ("OneHotEncoder", preprocessor),
        ("PCA", PCA()),
        ("StandardScaler", StandardScaler()),
        ("SelectKBest", SelectKBest(score_func=f_classif)),
        ("SVC", SVC(random_state=42)),
    ]
)


# ==============================================================================
# Paso 4. Optimización de Hiperparámetros con GridSearchCV
# ==============================================================================
# Malla de parámetros ajustada para asegurar altos scores sin sobrepasar tiempos
param_grid = {
    "SelectKBest__k": [20, 23],
    "SVC__C": [10],
    "SVC__gamma": ["scale"],
    "SVC__kernel": ["rbf"],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

grid_search.fit(x_train, y_train)


# ==============================================================================
# Paso 5. Guardar modelo comprimido en gzip
# ==============================================================================
os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(grid_search, file)


# ==============================================================================
# Pasos 6 y 7. Cálculo de Métricas y Matrices de Confusión
# ==============================================================================
y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

metrics_list = []

# Paso 6: Métricas cuantitativas de entrenamiento
metrics_list.append(
    {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    }
)

# Paso 6: Métricas cuantitativas de prueba
metrics_list.append(
    {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    }
)

# Paso 7: Matriz de confusión para entrenamiento
cm_train = confusion_matrix(y_train, y_train_pred)
metrics_list.append(
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }
)

# Paso 7: Matriz de confusión para prueba
cm_test = confusion_matrix(y_test, y_test_pred)
metrics_list.append(
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }
)

# Guardar en archivo JSON Lines
os.makedirs("files/output", exist_ok=True)
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for metric in metrics_list:
        f.write(json.dumps(metric) + "\n")