# Predicción de abandono de clientes

## Integrantes

- Abigail Justiniano

## Descripción

Este proyecto busca predecir si un cliente de una empresa de telecomunicaciones puede abandonar el servicio. Es un problema de clasificación binaria porque la variable `Churn` tiene dos posibles resultados: `Yes` o `No`.

Para esta primera entrega se trabajó con el dataset histórico. El archivo de producción no se utilizó porque queda reservado para una etapa posterior de la materia.

## Datos

El dataset histórico tiene 7.043 registros y 21 columnas. Durante la exploración encontramos:

- 5.186 clientes con `Churn = No` y 1.857 con `Churn = Yes`.
- 26 valores faltantes en `TotalCharges`.
- Variables numéricas y categóricas.
- `customerID` funciona como identificador, por eso se excluyó del entrenamiento.

Los datos se versionan con DVC y no se suben directamente a GitHub.

## Preprocesamiento

El preprocesamiento forma parte de un `Pipeline` de scikit-learn para que se apliquen los mismos pasos durante el entrenamiento y en usos posteriores.

- En las variables numéricas se completan faltantes con la mediana y luego se aplica `StandardScaler`.
- En las variables categóricas se completan faltantes con el valor más frecuente y se utiliza `OneHotEncoder`.
- La división entre entrenamiento y prueba usa `random_state=42` y `stratify=y` para conservar la proporción de las clases.

## Modelos comparados

Se comparan tres modelos:

1. `DummyClassifier`, usado como baseline.
2. `LogisticRegression`, como modelo lineal.
3. `RandomForestClassifier`, como modelo basado en árboles.

No se toma accuracy como única medida. Se comparan precision, recall, F1 y ROC-AUC. En este problema es importante mirar especialmente recall, porque un falso negativo significa considerar estable a un cliente que realmente iba a abandonar. En ese caso la empresa perdería la oportunidad de hacer una acción de retención.

## Resultados obtenidos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7942 | 0.6627 | 0.4489 | 0.5353 | 0.8120 |
| Random Forest | 0.7814 | 0.6416 | 0.3898 | 0.4849 | 0.7875 |
| Dummy Classifier | 0.7360 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

La regresión logística fue el modelo elegido porque consiguió el ROC-AUC más alto y también superó al random forest en recall y F1. El baseline obtuvo una accuracy que parece alta por el desbalance de clases, pero no detectó ningún caso positivo. Esto muestra por qué no alcanza con mirar solamente accuracy.

En el EDA también observamos que los clientes con contrato mes a mes tienen un porcentaje de abandono de 38,73%, mientras que en los contratos de un año es 14,97% y en los de dos años es 8,97%. Además, los clientes que abandonaron tenían en promedio menos antigüedad y cargos mensuales más altos.

## Estructura principal

```text
customer-churn-entrega1/
├── data/
├── metadata/
├── models/
├── reports/
├── src/
│   ├── data/eda.py
│   └── training/train.py
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows, para activar el entorno:

```bash
.venv\Scripts\activate
```

## Recuperar los datos

Después de clonar el repositorio y configurar el acceso al remote de DVC:

```bash
dvc pull
```

## Ejecutar el proyecto

Se puede ejecutar todo el flujo con:

```bash
dvc repro
```

También se pueden ejecutar las partes por separado:

```bash
python -m src.data.eda
python -m src.training.train
```

Los resultados quedan guardados en `reports/` y el pipeline seleccionado en `models/`.

## Decisiones y limitaciones

- Se utilizaron modelos simples porque son los solicitados para esta etapa y permiten comparar un baseline, un modelo lineal y uno de árboles.
- El modelo se selecciona inicialmente según ROC-AUC. También se revisan recall y F1 por el costo de no detectar a un cliente con intención de abandonar.
- No se utilizó el dataset de producción para entrenar ni mejorar el modelo.
- MLflow y Model Registry no se incluyeron porque todavía no fueron desarrollados en clase y se incorporarán en una entrega posterior.
