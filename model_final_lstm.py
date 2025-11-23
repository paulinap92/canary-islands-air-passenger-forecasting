# ==============================================================
# 🧠 Pronóstico final LSTM (con memoria secuencial, sin lags)
# ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------------------
# 1️⃣ PARÁMETROS
# --------------------------------------------------------------
ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
HORIZON_END = "2026-12-01"
WIN = 12  # número de meses en la memoria (ventana de secuencia)

# --------------------------------------------------------------
# 2️⃣ CARGA DE DATOS Y DEL MODELO
# --------------------------------------------------------------
df = pd.read_csv("result_total.csv", encoding="utf-8-sig")
df = df[df["Isla"] == ISLAND_NAME].copy()
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Características de calendario
if "month_sin" not in df.columns or "month_cos" not in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df[DATE_COL].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[DATE_COL].dt.month / 12)

if "year_norm" not in df.columns:
    base_year = df[DATE_COL].dt.year.min()
    df["year_norm"] = (df[DATE_COL].dt.year - base_year).astype(float) + 1.0

# ✅ Cargar modelo SIN compilación
model = load_model("models/lstm_best.h5", compile=False)
scaler_y = joblib.load("models/scaler_y.pkl")

# --------------------------------------------------------------
# 3️⃣ ESCALADO Y CREACIÓN DE LA SECUENCIA INICIAL
# --------------------------------------------------------------
# solo escalamos el target
y_scaled = scaler_y.transform(df[[TARGET_COL]])

# añadimos el target escalado como característica
df["_x_pasaj"] = y_scaled
FEAT_COLS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]
X_all = df[FEAT_COLS].to_numpy()

# últimos 12 meses (WIN) — memoria secuencial
seq = X_all[-WIN:].reshape(1, WIN, len(FEAT_COLS))

# --------------------------------------------------------------
# 4️⃣ GENERACIÓN DE FECHAS FUTURAS
# --------------------------------------------------------------
last_date = df[DATE_COL].max()
future_dates = pd.period_range(last_date, HORIZON_END, freq="M")[1:].to_timestamp()

df_future = df.copy()
base_year = df[DATE_COL].dt.year.min()

print(f"📅 Inicio: {last_date.date()} → Fin: {future_dates[-1].date()}")
print(f"🧠 Memoria de secuencia: {WIN} meses")

# --------------------------------------------------------------
# 5️⃣ PRONÓSTICO ITERATIVO
# --------------------------------------------------------------
for next_date in future_dates:
    month = next_date.month
    year = next_date.year
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    year_norm = (year - base_year) + 1.0

    # predicción en escala normalizada
    y_scaled_pred = float(model.predict(seq, verbose=0)[0][0])
    y_pred = float(scaler_y.inverse_transform([[y_scaled_pred]])[0][0])

    # añadir nueva fila
    df_future = pd.concat([df_future, pd.DataFrame([{
        "Isla": ISLAND_NAME,
        DATE_COL: next_date,
        TARGET_COL: y_pred,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "year_norm": year_norm,
        "_x_pasaj": y_scaled_pred
    }])], ignore_index=True)

    # actualización de la secuencia — desplazamiento de ventana
    next_step = np.array([[y_scaled_pred, month_sin, month_cos, year_norm]], dtype=float)
    seq = np.concatenate([seq[:, 1:, :], next_step.reshape(1, 1, -1)], axis=1)

# --------------------------------------------------------------
# 6️⃣ MARCAR FASE Y GRÁFICO
# --------------------------------------------------------------
df_future["Phase"] = np.where(df_future[DATE_COL] <= last_date, "History", "Forecast")

plt.figure(figsize=(10,5))
plt.plot(df_future[df_future["Phase"]=="History"][DATE_COL],
         df_future[df_future["Phase"]=="History"][TARGET_COL],
         label="Historia", color="tab:blue")
plt.plot(df_future[df_future["Phase"]=="Forecast"][DATE_COL],
         df_future[df_future["Phase"]=="Forecast"][TARGET_COL],
         label="Pronóstico LSTM", color="tab:green")
plt.title(f"✈️ {ISLAND_NAME} — Pronóstico LSTM (12 meses de memoria)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 7️⃣ GUARDAR RESULTADOS
# --------------------------------------------------------------
out_file = "forecast_total_canarias_lstm.csv"
df_future.to_csv(out_file, index=False, encoding="utf-8-sig")
print(f"💾 Guardado {out_file}")

print("\n📈 Últimos 12 meses del pronóstico:")
print(df_future[df_future["Phase"]=="Forecast"].tail(12)[[DATE_COL, TARGET_COL]])
