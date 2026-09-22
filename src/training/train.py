from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/customer_churn_historical.csv")
RESULTS_PATH = Path("reports/model_comparison.csv")
MODEL_PATH = Path("models/churn_pipeline.joblib")
RANDOM_STATE = 42


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, pipeline


def main():
    df = pd.read_csv(DATA_PATH)

    # customerID identifica al cliente, por eso no lo uso para entrenar el modelo
    x = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()

    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        metrics, trained_pipeline = evaluate_model(
            name, pipeline, x_train, x_test, y_train, y_test
        )
        results.append(metrics)
        trained_pipelines[name] = trained_pipeline

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results_df[metric_columns] = results_df[metric_columns].round(4)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    # se selecciona por ROC-AUC porque permite comparar la capacidad general
    # de separar clientes con y sin churn sin depender de un unico umbral.
    selected_name = results_df.iloc[0]["model"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_pipelines[selected_name], MODEL_PATH)

    print(results_df.to_string(index=False))
    print(f"\nModelo seleccionado: {selected_name}")
    print(f"Pipeline guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
