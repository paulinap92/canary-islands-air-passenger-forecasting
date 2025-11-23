# ==============================================================
# 🧭 Pronóstico final XGBoost — generación corregida de lags
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# === PARÁMETROS ===
ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
HORIZON_END = "2026-12-01"

# === Datos de entrada ===
df = pd.read_csv("result_total_with_lags_coded.csv", encoding="utf-8-sig")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)
df = df[df["Isla"] == ISLAND_NAME].reset_index(drop=True)

# 🔹 Tendencia a largo plazo
df["month_idx"] = np.arange(len(df))

# 🔹 Características para el modelo
FEATURES = [
    "month_idx", "month_sin", "month_cos", "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3", "roll6"
]

X_train = df[FEATURES].values
y_train = df[TARGET_COL].values
print(y_train)

# 🔹 Modelo XGB
xgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    ))
])

model = TransformedTargetRegressor(regressor=xgb, transformer=StandardScaler())
model.fit(X_train, y_train)
print("✅ Modelo entrenado con datos históricos")

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
    min_year = df["Año"].min()
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
print("💾 Guardado forecast_total_canarias_xgb.csv")

print("\n📈 Últimos 12 meses del pronóstico:")
