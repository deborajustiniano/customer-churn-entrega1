# Predicción de abandono de clientes

## Integrante

- Abigail Justiniano

## Descripción

En este proyecto trabajé con datos de clientes de una empresa de telecomunicaciones. El objetivo es predecir si un cliente va a abandonar el servicio o no, usando la variable `Churn`.

Para esta primera entrega utilicé solamente el archivo histórico, ya que es el que contiene el resultado real de `Churn`. Los archivos de producción y scoring quedan para una etapa posterior.

## Análisis de los datos

El dataset histórico tiene 7.043 filas y 21 columnas.

Durante el análisis encontré lo siguiente:

- 5.186 clientes no abandonaron el servicio.
- 1.857 clientes sí abandonaron el servicio.
- Hay 26 valores faltantes en la columna `TotalCharges`.
- `customerID` se usa solamente para identificar a cada cliente, por eso no se incluyó en el entrenamiento.
- La cantidad de clientes que no abandonan es mayor que la cantidad que sí lo hace.

También observé que el abandono cambia según el tipo de contrato:

- Contrato mensual: 38,73 % de abandono.
- Contrato de un año: 14,97 %.
- Contrato de dos años: 8,97 %.

Los clientes que abandonaron también tenían, en promedio, menos antigüedad y cargos mensuales más altos.

## Preparación de los datos

Separé las columnas numéricas de las categóricas para poder tratarlas de manera diferente.

Para las columnas numéricas:

- Completé los valores faltantes usando la mediana.
- Apliqué `StandardScaler` para escalar los valores.

Para las columnas categóricas:

- Completé los faltantes usando el valor más frecuente.
- Apliqué `OneHotEncoder` para convertir las categorías en valores que puedan usar los modelos.

Todo este procesamiento se encuentra dentro de un pipeline de scikit-learn. De esta manera, se aplican siempre los mismos pasos antes de usar el modelo.

Los datos se dividieron en entrenamiento y prueba. Usé `random_state=42` para poder repetir el resultado y `stratify=y` para conservar aproximadamente la misma proporción de clientes con y sin churn.

## Modelos probados

Probé tres modelos:

1. `DummyClassifier`, como resultado básico de referencia.
2. `LogisticRegression`.
3. `RandomForestClassifier`.

No comparé los modelos solamente por accuracy, porque el dataset tiene más clientes sin abandono que con abandono. También revisé precision, recall, F1 y ROC-AUC.

## Resultados

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7942 | 0.6627 | 0.4489 | 0.5353 | 0.8120 |
| Random Forest | 0.7864 | 0.6578 | 0.3978 | 0.4958 | 0.7896 |
| Dummy Classifier | 0.7360 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

El modelo elegido fue regresión logística, porque tuvo el ROC-AUC más alto y también obtuvo mejores resultados de recall y F1 que random forest.

El resultado del `DummyClassifier` muestra por qué no conviene mirar solamente accuracy. Aunque consiguió 73,60 % de accuracy, no detectó ningún cliente que fuera a abandonar el servicio.

El recall también es importante en este caso, porque un falso negativo representa a un cliente que el modelo considera estable, pero que en realidad va a abandonar. Esto puede hacer que la empresa no llegue a realizar una acción de retención a tiempo.

## Estructura del proyecto

```text
customer-churn-entrega1/
├── data/
│   └── raw/
├── metadata/
├── models/
├── reports/
├── src/
│   ├── data/
│   │   └── eda.py
│   └── training/
│       └── train.py
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Instalación

Crear y activar el entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Descargar los datos

Los datos y los resultados generados se versionan con DVC y se almacenan en DagsHub. No se suben directamente a GitHub.

Después de clonar el repositorio y configurar el acceso al almacenamiento remoto, se pueden descargar con:

```bash
dvc pull
```

## Ejecutar el proyecto

Para ejecutar todo el flujo:

```bash
dvc repro
```

También se pueden ejecutar las etapas por separado:

```bash
python -m src.data.eda
python -m src.training.train
```

El análisis exploratorio queda guardado en `reports/eda_summary.txt`, la comparación de los modelos en `reports/model_comparison.csv` y el pipeline seleccionado en `models/churn_pipeline.joblib`.

## Uso de DVC

DVC se utilizó para:

- Versionar el dataset histórico.
- Definir las etapas de análisis y entrenamiento.
- Guardar el modelo y los resultados.
- Poder reproducir el proyecto con `dvc repro`.
- Almacenar los archivos pesados en DagsHub.

## Limitaciones

En esta entrega se hizo una primera comparación con modelos simples. Todavía no se realizó ajuste de hiperparámetros ni se probó el modelo con datos nuevos de producción.

Tampoco se incorporó MLflow porque todavía no fue trabajado en clase y queda para una entrega posterior.
