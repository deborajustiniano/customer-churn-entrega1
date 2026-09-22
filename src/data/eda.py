from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_churn_historical.csv")
REPORT_PATH = Path("reports/eda_summary.txt")


def main():
    df = pd.read_csv(DATA_PATH)

    target_counts = df["Churn"].value_counts()
    target_percentages = (df["Churn"].value_counts(normalize=True) * 100).round(2)
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    summary = [
        "RESUMEN DEL ANÁLISIS EXPLORATORIO",
        "",
        f"Cantidad de filas: {df.shape[0]}",
        f"Cantidad de columnas: {df.shape[1]}",
        "",
        "Distribución de Churn:",
        target_counts.to_string(),
        "",
        "Porcentaje de Churn:",
        target_percentages.to_string(),
        "",
        "Valores faltantes:",
        missing_values.to_string() if not missing_values.empty else "No se encontraron faltantes.",
        "",
        "Promedios de variables numéricas por Churn:",
        df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]]
        .mean()
        .round(2)
        .to_string(),
        "",
        "Distribución de Churn según tipo de contrato:",
        pd.crosstab(df["Contract"], df["Churn"], normalize="index")
        .mul(100)
        .round(2)
        .to_string(),
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(summary), encoding="utf-8")
    print(f"EDA terminado. Resultado guardado en {REPORT_PATH}")


if __name__ == "__main__":
    main()
