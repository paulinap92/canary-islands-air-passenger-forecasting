# ==============================================================
# 🧭 Pronóstico final XGBoost — uso del modelo guardado
# ==============================================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === PARÁMETROS ===
ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
HORIZON_END = "2026-12-01"
MODEL_PATH = Path("models/xgb_best.pkl")

# === Datos de entrada ===
df = pd.read_csv("result_total_with_lags_coded.csv", encoding="utf-8-sig")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)
df = df[df["Isla"] == ISLAND_NAME].reset_index(drop=True)

# 🔹 Tendencia a largo plazo
df["month_idx"] = np.arange(len(df))

# 🔹 Características esperadas por el modelo guardado
FEATURES = [
    "month_sin", "month_cos", "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3", "roll6"
]

# 🔹 Cargar el modelo existente; este archivo no entrena modelos
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo guardado: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
expected_features = getattr(model, "n_features_in_", None)
if expected_features is not None and expected_features != len(FEATURES):
    raise ValueError(
        f"El modelo guardado espera {expected_features} variables, "
        f"pero el pronóstico proporciona {len(FEATURES)}."
    )

print(f"✅ Modelo cargado desde {MODEL_PATH}")

# ==============================================================
# Pronóstico iterativo — lags corregidos
# ==============================================================
df_future = df.copy()
last_date = df_future[DATE_COL].max()
future_dates = pd.period_range(last_date, HORIZON_END, freq="M")[1:].to_timestamp()

print(f"📈 Pronosticando desde {last_date.date()} hasta {future_dates[-1].date()}")

for next_date in future_dates:
    new_row = {}

    # --- características de calendario
    new_row[DATE_COL] = next_date
    new_row["Isla"] = ISLAND_NAME
    m = next_date.month
    new_row["month_sin"] = np.sin(2 * np.pi * m / 12)
    new_row["month_cos"] = np.cos(2 * np.pi * m / 12)
    new_row["month_idx"] = len(df_future)
    min_year = df[DATE_COL].dt.year.min()

    new_row["year_norm"] = (next_date.year - min_year) + 1

    # --- lags: desde los últimos meses en df_future
    for i in range(1, 13):
        if len(df_future) >= i:
            new_row[f"lag_{i}"] = df_future[TARGET_COL].iloc[-i]
        else:
            new_row[f"lag_{i}"] = np.nan

    # --- rollings
    last_vals = df_future[TARGET_COL].tail(6).values
    new_row["roll3"] = np.mean(last_vals[-3:]) if len(last_vals) >= 3 else np.nan
    new_row["roll6"] = np.mean(last_vals[-6:]) if len(last_vals) >= 6 else np.nan

    X_pred = np.array([[new_row.get(f, np.nan) for f in FEATURES]])
    y_pred = model.predict(X_pred)[0]

    # --- sin valores negativos
    y_pred = max(y_pred, 0)
    new_row[TARGET_COL] = y_pred

    df_future = pd.concat([df_future, pd.DataFrame([new_row])], ignore_index=True)

df_future["Phase"] = np.where(df_future[DATE_COL] <= last_date, "History", "Forecast")

# ==============================================================
# 📊 Gráfico
# ==============================================================
plt.figure(figsize=(10,5))
plt.plot(df_future[df_future["Phase"]=="History"][DATE_COL],
         df_future[df_future["Phase"]=="History"][TARGET_COL],
         label="Historia", color="tab:blue")
plt.plot(df_future[df_future["Phase"]=="Forecast"][DATE_COL],
         df_future[df_future["Phase"]=="Forecast"][TARGET_COL],
         label="Pronóstico XGB (lags corregidos)", color="tab:orange")
plt.title("✈️ Total Canarias — Pronóstico XGB hasta 2026 (versión corregida)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================
# 📁 Guardar resultados
# ==============================================================
df_future.to_csv("forecast_total_canarias_xgb.csv", index=False, encoding="utf-8-sig")
print("💾 Guardado forecast_total_canarias_fixedlags.csv")

print("\n📈 Últimos 12 meses del pronóstico:")
print(df_future[df_future["Phase"] == "Forecast"].tail(12)[[DATE_COL, TARGET_COL]])
