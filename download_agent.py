import os
import re
import shutil
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ==============================================================
# 🔹 Constantes / configuración
# ==============================================================
DATA_DIR = "data"
BACKUP_DIR = "backup_results"
RESULT_DETAILS_CSV = "result.csv"
RESULT_TOTAL_CSV = "result_total.csv"
FEATURES_WITH_LAGS = "result_total_with_lags.csv"
FEATURES_WITH_LAGS_CODED = "result_total_with_lags_coded.csv"

BASE_URL = "https://www.webtenerife.com/investigacion/situacion-turistica/trafico-aereo/"
SPANISH_MONTHS = [
    "enero","febrero","marzo","abril","mayo","junio",
    "julio","agosto","septiembre","octubre","noviembre","diciembre"
]
MESES_MAP = {m:i+1 for i, m in enumerate(SPANISH_MONTHS)}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


# ==============================================================
# 🔹 Utilidades
# ==============================================================
def parse_month_year_from_filename(filename: str):
    """
    Busca MES + AÑO en el nombre del archivo sin importar el separador (_ - espacio).
    Devuelve: (mes:str, año:int) por ejemplo ("agosto", 2025)
    """
    stem = Path(filename).stem.lower()
    tokens = re.split(r"[_\-\s]+", stem)
    month_idxs = [i for i, t in enumerate(tokens) if t in SPANISH_MONTHS]
    if not month_idxs:
        raise ValueError(f"No se encontró un mes en el nombre del archivo: {filename}")

    for i in reversed(month_idxs):
        if i + 1 < len(tokens) and re.fullmatch(r"\d{4}", tokens[i+1]):
            return tokens[i], int(tokens[i+1])

    for i in reversed(month_idxs):
        for j in range(i+1, len(tokens)):
            if re.fullmatch(r"\d{4}", tokens[j]):
                return tokens[i], int(tokens[j])

    raise ValueError(f"No fue posible emparejar mes+año en el nombre del archivo: {filename}")


def normalize_isla_name(value: str):
    if not isinstance(value, str):
        return ""
    value = value.strip()
    value = re.sub(r"\s*\(.*?\)", "", value)
    return value


def load_and_clean_excel(file_path: str) -> pd.DataFrame:
    """Limpia y carga la hoja con datos de pasajeros."""
    excel_preview = pd.read_excel(file_path, header=None)
    header_line = excel_preview[
        excel_preview.apply(
            lambda r: r.astype(str).str.contains("AEROPUERTO PROCEDENCIA", case=False, na=False)
        ).any(axis=1)
    ].index
    df = pd.read_excel(file_path, skiprows=header_line[0], header=None)

    # eliminar columnas/filas vacías y registros basura/pies de página
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").replace(["-", ""], pd.NA)
    df.columns = df.columns.astype(str).str.strip()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()

    patterns = [
        r"LLEGADA\s+DE\s+PASAJEROS",
        r"REGULAR\s*\+\s*NO\s*REGULAR",
        r"FUENTE", r"ELABORACI", r"TURISMO\s+DE\s+TENERIFE"
    ]
    combined = "(" + "|".join(patterns) + ")"
    df = df[~df.astype(str).apply(lambda x: x.str.contains(combined, case=False, na=False, regex=True)).any(axis=1)]

    return df.dropna(how="all").reset_index(drop=True)


def split_islands_from_combined_table(df: pd.DataFrame):
    """
    A partir de una hoja grande con varias islas crea la lista de marcadores de islas.
    """
    islas = ["GRAN CANARIA", "FUERTEVENTURA", "LANZAROTE", "TENERIFE", "LA PALMA", "TOTAL CANARIAS"]
    positions = []
    for r_idx, row in df.iterrows():
        for c_idx, value in enumerate(row):
            if normalize_isla_name(value) in islas:
                positions.append({"isla": normalize_isla_name(value), "row": r_idx, "col": c_idx})
    positions.sort(key=lambda p: (p["row"], p["col"]))
    return positions


def extract_isla_data(df, positions, nrows=40):
    """
    Crea un DataFrame para cada isla, combinando etiquetas y valores.
    Devuelve dict: { "TENERIFE": df_isla, ... }
    """
    results = {}
    for pos in positions:
        isla = pos['isla']
        row = pos['row']
        col = pos['col']

        labels = df.iloc[row + 2: row + 2 + nrows, 0].reset_index(drop=True)
        values = df.iloc[row + 2: row + 2 + nrows, col: col + 5].reset_index(drop=True)

        df_isla = pd.concat([labels, values], axis=1)
        df_isla.columns = ['AEROPUERTO_DE_PROCEDENCIA'] + list(df.iloc[row + 1, col: col + 5])
        df_isla['Isla'] = isla.title()

        results[isla] = df_isla
    return results


def backup_current_results():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(RESULT_DETAILS_CSV).exists():
        shutil.copy(RESULT_DETAILS_CSV, Path(BACKUP_DIR) / f"result_{ts}.csv")
    if Path(RESULT_TOTAL_CSV).exists():
        shutil.copy(RESULT_TOTAL_CSV, Path(BACKUP_DIR) / f"result_total_{ts}.csv")


# ==============================================================
# 🔹 Procesamiento principal del archivo
# ==============================================================
def process_new_excel(file_path, results_details, results_total, nrows=40):
    """
    Procesa un nuevo archivo Excel y añade registros a dos tablas:
      - result.csv  (detalles: cada categoría/origen)
      - result_total.csv (agregado 'TOTAL PASAJEROS' por isla)
    Guardado precedido por conversión numérica y dropna.
    """
    file_path = Path(file_path)
    file_name = file_path.stem

    # 1) mes/año desde el nombre
    mes, anio = parse_month_year_from_filename(file_name)
    mes_num = MESES_MAP.get(mes)
    if not mes_num:
        raise ValueError(f"Mes desconocido: {mes}")
    fecha = datetime(int(anio), mes_num, 1)
    fecha_str = fecha.strftime("%Y-%m-%d")
    target_col = f"{mes} {anio}"

    # 2) cargar y extraer secciones de islas
    df_raw = load_and_clean_excel(str(file_path))
    positions = split_islands_from_combined_table(df_raw)
    df_islas = extract_isla_data(df_raw, positions, nrows=nrows)

    # 3) construir listas de registros
    details_list, total_list = [], []

    for isla, dfi in df_islas.items():
        dfi["Isla"] = isla.title()

        if target_col not in dfi.columns:
            print(f"⚠️ Falta la columna '{target_col}' en los datos de {isla}")
            continue

        # detalles
        df_detail = dfi[["AEROPUERTO_DE_PROCEDENCIA", target_col]].copy()
        df_detail["Mes_Año"] = target_col
        df_detail["Pasajeros"] = pd.to_numeric(df_detail[target_col], errors="coerce")
        df_detail["Mes"] = mes
        df_detail["Año"] = int(anio)
        df_detail["MesNum"] = int(mes_num)
        df_detail["Fecha"] = fecha_str
        df_detail["Isla"] = isla.title()
        df_detail = df_detail[
            ["AEROPUERTO_DE_PROCEDENCIA", "Mes_Año", "Pasajeros", "Isla", "Mes", "Año", "MesNum", "Fecha"]
        ]
        details_list.append(df_detail)

        # TOTAL PASAJEROS (fila exacta)
        mask_total = (
            dfi["AEROPUERTO_DE_PROCEDENCIA"].astype(str).str.strip().str.lower().eq("total pasajeros")
        )
        df_total = dfi.loc[mask_total, [target_col]].copy()

        if not df_total.empty:
            total_value = pd.to_numeric(df_total[target_col].iloc[0], errors="coerce")
            total_list.append({
                "Isla": isla.title(),
                "Fecha": fecha_str,
                "Mes": mes,
                "Año": int(anio),
                "MesNum": int(mes_num),
                "Pasajeros": total_value
            })
        else:
            print(f"⚠️ No se encontró la fila 'TOTAL PASAJEROS' para {isla}")

    # 4) concatenar → limpiar → deduplicar
    df_details = pd.concat(details_list, ignore_index=True) if details_list else pd.DataFrame()
    df_totals = pd.DataFrame(total_list)

    for df in (df_details, df_totals):
        if not df.empty:
            if "Fecha" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            if "Pasajeros" in df.columns:
                df["Pasajeros"] = pd.to_numeric(df["Pasajeros"], errors="coerce")

    # dejar solo registros completos
    if not df_details.empty:
        df_details = df_details.dropna(subset=["Fecha", "Pasajeros", "AEROPUERTO_DE_PROCEDENCIA", "Isla", "Año", "MesNum"])
        df_details = df_details.drop_duplicates(
            subset=["Isla", "Fecha", "AEROPUERTO_DE_PROCEDENCIA"], keep="last"
        )

    if not df_totals.empty:
        df_totals = df_totals.dropna(subset=["Fecha", "Pasajeros", "Isla", "Año", "MesNum"])
        df_totals = df_totals.drop_duplicates(subset=["Isla", "Fecha"], keep="last")

    # 5) añadir a las tablas existentes y guardar SIN vacíos
    if not results_details.empty:
        results_details["Fecha"] = pd.to_datetime(results_details.get("Fecha"), errors="coerce")
        results_details["Pasajeros"] = pd.to_numeric(results_details.get("Pasajeros"), errors="coerce")

    if not results_total.empty:
        results_total["Fecha"] = pd.to_datetime(results_total.get("Fecha"), errors="coerce")
        results_total["Pasajeros"] = pd.to_numeric(results_total.get("Pasajeros"), errors="coerce")

    results_details = pd.concat([results_details, df_details], ignore_index=True) if not df_details.empty else results_details
    results_total = pd.concat([results_total, df_totals], ignore_index=True) if not df_totals.empty else results_total

    # limpieza global
    if not results_details.empty:
        results_details = results_details.dropna(subset=["Fecha", "Pasajeros"]).sort_values(["Isla", "Fecha"])
        results_details = results_details.drop_duplicates(
            subset=["Isla", "Fecha", "AEROPUERTO_DE_PROCEDENCIA"], keep="last"
        )

    if not results_total.empty:
        results_total = results_total.dropna(subset=["Fecha", "Pasajeros"]).sort_values(["Isla", "Fecha"])
        results_total = results_total.drop_duplicates(subset=["Isla", "Fecha"], keep="last")

    # 6) guardado (después de dropna)
    results_details.to_csv(RESULT_DETAILS_CSV, index=False, encoding="utf-8-sig")
    results_total.to_csv(RESULT_TOTAL_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ Guardado {RESULT_DETAILS_CSV} y {RESULT_TOTAL_CSV} (después de dropna + deduplicación).")
    return results_details, results_total


# ==============================================================
# 🔹 Construcción de características (lags/rolling) – paso separado
# ==============================================================
def build_features(result_total_csv=RESULT_TOTAL_CSV,
                   out_with_lags=FEATURES_WITH_LAGS,
                   out_with_lags_coded=FEATURES_WITH_LAGS_CODED):
    """
    Calcula características cíclicas, lags 1..12 y rolling 3/6 para cada isla.
    No se ejecuta automáticamente en run(); ejecútalo cuando quieras (p. ej. en CI).
    """
    import numpy as np

    if not Path(result_total_csv).exists():
        print(f"⚠️ Falta {result_total_csv} – omito build_features.")
        return

    df = pd.read_csv(result_total_csv, encoding="utf-8-sig")
    if df.empty:
        print("⚠️ result_total.csv vacío – nada que hacer.")
        return

    # sanity + tipos
    for col in ("Fecha", "Pasajeros", "Isla", "Año", "MesNum"):
        if col not in df.columns:
            raise KeyError(f"Falta la columna '{col}' en {result_total_csv}")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Pasajeros"] = pd.to_numeric(df["Pasajeros"], errors="coerce")
    df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
    df["MesNum"] = pd.to_numeric(df["MesNum"], errors="coerce")

    df = df.dropna(subset=["Fecha", "Pasajeros", "Isla", "Año", "MesNum"]).copy()
    df = df.sort_values(["Isla", "Fecha"]).reset_index(drop=True)

    # características cíclicas y year_norm
    df["month_sin"] = np.sin(2 * np.pi * df["MesNum"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["MesNum"] / 12.0)
    years_sorted = sorted(df["Año"].astype(int).unique())
    year_map = {y: i + 1 for i, y in enumerate(years_sorted)}
    df["year_norm"] = df["Año"].astype(int).map(year_map)

    # lags 1..12 por isla
    fe = df.copy()
    for lag in range(1, 13):
        fe[f"lag_{lag}"] = fe.groupby("Isla")["Pasajeros"].shift(lag)

    # rolling (sobre desplazado 1)
    s1 = fe.groupby("Isla")["Pasajeros"].shift(1)
    fe["roll3"] = s1.groupby(fe["Isla"]).rolling(3).mean().reset_index(level=0, drop=True)
    fe["roll6"] = s1.groupby(fe["Isla"]).rolling(6).mean().reset_index(level=0, drop=True)

    lag_cols = [f"lag_{i}" for i in range(1, 13)] + ["roll3", "roll6"]
    fe_full = fe.dropna(subset=lag_cols).reset_index(drop=True)

    # guardado
    fe.to_csv(out_with_lags, index=False, encoding="utf-8-sig")
    fe_full.to_csv(out_with_lags_coded, index=False, encoding="utf-8-sig")
    print(f"📈 Guardadas características: {out_with_lags} y {out_with_lags_coded}")


# ==============================================================
# 🔹 Agente
# ==============================================================
class PassengerAgent:
    def __init__(self, data_dir=DATA_DIR, base_url=BASE_URL):
        self.data_dir = data_dir
        self.base_url = base_url

    def get_last_month_file(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".xlsx")]
        if not files:
            return None
        return sorted(files)[-1]

    def get_next_month_name(self, last_file: str):
        mes, anio = parse_month_year_from_filename(last_file)
        idx = SPANISH_MONTHS.index(mes)
        next_idx = (idx + 1) % 12
        next_year = anio + 1 if next_idx == 0 else anio
        return SPANISH_MONTHS[next_idx], next_year

    def download_latest_file(self, target_month: str, target_year: int):
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.1 Safari/537.36"),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://www.google.com/",
        }

        print(f"🌍 Cargando página: {self.base_url}")
        resp = requests.get(self.base_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = [a["href"] for a in soup.find_all("a", href=True) if ".xlsx" in a["href"].lower()]
        links = [urljoin(self.base_url, href) for href in links]

        want_month = target_month.lower()
        want_year = str(target_year)
        cand = [u for u in links if want_month in u.lower() and want_year in u]

        if not cand:
            print("⚠️ Archivos disponibles (fragmento):")
            for u in links[-10:]:
                print("  -", os.path.basename(u))
            raise RuntimeError(f"No se encontró archivo para: {want_month} {want_year}")

        excel_url = sorted(cand)[-1]

        existing = [f for f in os.listdir(self.data_dir) if f.endswith(".xlsx")]
        next_seq = 1
        if existing:
            nums = []
            for f in existing:
                m = re.match(r"^(\d+)_", f)
                if m:
                    nums.append(int(m.group(1)))
            next_seq = max(nums) + 1 if nums else 1

        filename = f"{next_seq:02d}_pasajeros_canarias_{want_month}_{want_year}.xlsx"
        dst = os.path.join(self.data_dir, filename)

        if os.path.exists(dst):
            print(f"ℹ️ Archivo ya existe, omito descarga: {dst}")
            return dst

        fresp = requests.get(excel_url, headers=headers)
        fresp.raise_for_status()
        with open(dst, "wb") as f:
            f.write(fresp.content)

        print(f"📦 Guardado: {dst}")
        return dst

    def run(self):
        """
        Ciclo principal: encuentra el nuevo mes, descarga el archivo,
        haz backup, actualiza CSV (con DROPNAs), FIN.
        (Sin lags en este paso — lo hace build_features()).
        """
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".xlsx")])
        if not files:
            raise RuntimeError("❌ No hay archivos en data/. Añade el primero manualmente.")

        last_file = files[-1]
        next_month, next_year = self.get_next_month_name(last_file)
        print(f"🧭 Último archivo: {last_file} → Busco: {next_month.title()} {next_year}")

        # 1) descargar nuevo archivo
        new_path = self.download_latest_file(next_month, next_year)
        print(f"✅ Nuevo archivo guardado: {new_path}")

        # 2) cargar CSV existentes
        if not Path(RESULT_DETAILS_CSV).exists() or not Path(RESULT_TOTAL_CSV).exists():
            raise FileNotFoundError("❌ Faltan result.csv o result_total.csv. Necesitas datos base.")

        results_details = pd.read_csv(RESULT_DETAILS_CSV, encoding="utf-8-sig")
        results_total = pd.read_csv(RESULT_TOTAL_CSV, encoding="utf-8-sig")

        # 3) backup
        backup_current_results()
        print("💾 Backup OK.")

        # 4) procesar nuevo archivo → actualización CSV (dropna dentro)
        before_rows = len(results_total)
        results_details, results_total = process_new_excel(new_path, results_details, results_total)
        added_rows = len(results_total) - before_rows
        print(f"➕ Añadidos {added_rows} registros a {RESULT_TOTAL_CSV}")

        # 5) (opcional) reentrenamiento después de actualizar
        if os.getenv("RUN_RETRAIN") == "1":
            try:
                print("🚀 RUN_RETRAIN=1 → inicio reentrenamiento de modelos...")
                os.system("python model_final_xgb.py")
                os.system("python model_final_lstm.py")
                print("✅ Modelos reentrenados.")
            except Exception as e:
                print(f"⚠️ Error durante el entrenamiento: {e}")


# ==============================================================
# 🔹 Lanzamiento
# ==============================================================
if __name__ == "__main__":
    agent = PassengerAgent()
    agent.run()
    # Si quieres construir las características para XGB/LSTM de inmediato, descomenta:
    # build_features()
