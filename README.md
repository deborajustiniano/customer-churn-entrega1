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
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   └── eda.py
│   └── training/
│       └── train.py
├── tests/
│   ├── conftest.py
│   └── test_inference.py
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

## Segundo parcial: experimentos con MLflow

En esta etapa agregué MLflow al entrenamiento de la primera entrega. Se mantienen
los tres modelos, la división de datos, la semilla 42 y la selección por ROC-AUC.
Uso MLflow 2.22.2, compatible con Python 3.9.

Cada ejecución guarda el nombre y los parámetros del modelo, las cinco métricas
y el pipeline completo como artefacto `pipeline`. El pipeline ganador se registra
en el Model Registry de DagsHub con el nombre `customer-churn`. Cada entrenamiento
completo crea tres ejecuciones y una nueva versión del modelo seleccionado.
El archivo `models/churn_pipeline.joblib` sigue siendo la copia para la futura API.

Cada pipeline incluye una `signature`, inferida con las 19 columnas originales
del conjunto de prueba y sus predicciones. Describe los nombres y tipos de las
entradas y el tipo de salida: una clase `0` (sin abandono) o `1` (con abandono)
por cliente. El `input_example` contiene tres filas reales de prueba, sin
`customerID` ni `Churn`, antes del preprocesamiento. La futura API deberá enviar
esas mismas columnas al pipeline; el contrato de MLflow no reemplaza la validación
de categorías y rangos que se agregará en esa etapa.

### Configurar DagsHub

Actualizar las dependencias y crear la configuración local:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Completar `.env` localmente con la URI del repositorio en DagsHub, el usuario
y un token de acceso en `MLFLOW_TRACKING_PASSWORD`. No compartir ese archivo
ni agregarlo a Git: `.env` ya está ignorado. Los valores del ejemplo son ficticios.
Si un valor contiene caracteres especiales del shell, escribirlo entre comillas simples.

Cargar las variables en la terminal antes de entrenar:

```bash
set -a
source .env
set +a
python -m src.training.train
```

El script no carga `.env` automáticamente y avisa si falta alguna de las tres
variables. MLflow usa la misma URI para tracking y Model Registry; no requiere
levantar un servidor local. Las credenciales de DVC se configuran por separado.

También se puede usar `dvc repro` con las variables cargadas. Como DVC evita
repetir etapas sin cambios, para crear nuevas ejecuciones de MLflow se puede
ejecutar directamente el módulo de entrenamiento o usar `dvc repro --force train`.
Después se pueden consultar las ejecuciones y la versión registrada en la sección
de experimentos y modelos del repositorio en DagsHub.

Referencia: [integración de DagsHub con MLflow](https://dagshub.com/docs/integration_guide/mlflow_tracking/).

### Advertencias numéricas del entorno

En el entorno revisado (macOS ARM, Python 3.9 y NumPy 2.0.2 con BLAS Accelerate)
aparecen advertencias de división por cero, overflow y valores inválidos durante
operaciones matriciales. También se reprodujeron con matrices aleatorias pequeñas,
sin entrenar ningún modelo. El resultado fue finito y coincidió con un cálculo
alternativo con una diferencia máxima de aproximadamente `1e-14`.

El histórico no tiene infinitos numéricos; los 26 faltantes de `TotalCharges`
son los previstos y el pipeline los imputa. La causa probable está en la combinación
NumPy/Accelerate del entorno, aunque no se confirmó el origen exacto. Por eso
no cambié el modelo, el preprocesamiento ni oculté las advertencias.

## Segundo parcial: API de inferencia y pruebas

La API usa el pipeline completo de `models/churn_pipeline.joblib` y lo carga
una sola vez al iniciar. No entrena ni se conecta con DagsHub. Si el archivo
falta o no se puede cargar, el inicio falla con un mensaje que explica el problema.

Con el entorno virtual activado, ejecutar desde la raíz del proyecto:

```bash
pip install -r requirements.txt
# Si todavía no está el modelo, recuperarlo con el acceso DVC configurado:
dvc pull
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

La documentación interactiva está en `http://127.0.0.1:8000/docs`.
`GET /health` devuelve HTTP 200 con `{"status":"ok","model_loaded":true}`.

`POST /predict` recibe un solo cliente con estas 19 variables:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 50.0,
  "TotalCharges": 600.0
}
```

Se puede pegar el JSON en `/docs`, usando “Try it out”. La respuesta contiene
`prediction` (`Yes` o `No`) y `probability`, la probabilidad de abandono (`Yes`)
entre 0 y 1, incluso cuando la clase predicha es `No`.

Pydantic valida la entrada y la salida con modelos separados. Todos los campos
son obligatorios, aunque `TotalCharges` puede ser `null`: se convierte en `NaN`
para que el pipeline aplique la imputación. Los números deben enviarse como
números, sin comillas; `SeniorCitizen` admite 0 o 1, `tenure` es entero no negativo
y los cargos son no negativos y finitos. Se admiten las categorías del diccionario
de datos. Se rechazan campos adicionales, incluidos `customerID` y `Churn`.
Las entradas inválidas devuelven HTTP 422. El DataFrame se ordena según las
columnas guardadas en el pipeline, independientemente del orden del JSON.

Para ejecutar las pruebas:

```bash
python -m pytest -q
```

Las pruebas usan `TestClient` y el modelo real descargado; no necesitan variables
MLflow, credenciales ni conexión de red. Cubren el contrato de datos, fit y
transform del preprocesador con datos pequeños, carga del modelo, `/predict`,
entradas inválidas, inferencia de extremo a extremo (también con `TotalCharges`
nulo), `/health` y `/docs`. Si el modelo todavía no está, hay que recuperarlo
antes con DVC. Las advertencias numéricas del entorno descritas arriba también
pueden aparecer al predecir.

Referencias: [inicio de la aplicación con lifespan](https://fastapi.tiangolo.com/advanced/events/)
y [validación estricta con Pydantic](https://docs.pydantic.dev/latest/concepts/strict_mode/).

## Ejecutar la API con Docker Compose

Se necesita Docker con Docker Compose disponible y el modelo local **antes de
construir la imagen**. Desde la raíz del proyecto, con el entorno virtual activo
y el acceso DVC configurado, ejecutar:

```bash
dvc pull
docker compose up --build -d
docker compose ps
```

El primer comando recupera `models/churn_pipeline.joblib`; sin ese archivo falla
la construcción. Las credenciales se usan solamente en la computadora para DVC:
no se pasan al contenedor. No es necesario cargar las variables de MLflow para
ejecutar la API. Si hay otra API local en el puerto 8000, detenerla antes.

Compose levanta un único servicio `api`, basado en `python:3.9-slim`. Instala
`requirements-api.txt` y copia únicamente el código de la API y el pipeline local.
`.dockerignore` limita el contexto para excluir `.env`, `.dvc`, `.git`, datos,
cachés y entornos virtuales. No se ejecuta entrenamiento dentro del contenedor.

`requirements.txt` se usa para desarrollo, entrenamiento y pruebas locales.
`requirements-api.txt` contiene solamente las dependencias de inferencia del
contenedor, sin DVC, S3, MLflow ni herramientas de pruebas. Sus versiones se
tomaron del entorno actual, incluidas NumPy, SciPy y threadpoolctl que utiliza
scikit-learn, para mantener la compatibilidad con el pipeline entrenado.

El healthcheck consulta `/health` cada 10 segundos y comprueba que el modelo
esté cargado. Esperar a que `docker compose ps` muestre `healthy`. También se
puede iniciar y esperar desde la terminal con `docker compose up --build -d --wait`.

Para consultar el estado:

```bash
curl http://localhost:8000/health
```

Debe responder `{"status":"ok","model_loaded":true}`. Abrir
`http://localhost:8000/docs` y usar el ejemplo de cliente de la sección anterior
para probar `POST /predict` desde Swagger UI.

Para detener y eliminar los contenedores y la red de este Compose:

```bash
docker compose down
```

Este comando se ejecuta desde este proyecto, sin opciones para borrar imágenes
ni volúmenes. No afecta los contenedores de otros proyectos Compose.
