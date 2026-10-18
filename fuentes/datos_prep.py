"""
datos_prep.py — Funciones de limpieza y estandarización del dataset AI4I 2020.

Proyecto: Actividad 5 · Módulo predictivo de proceso para VisionQ
Autor: Rene

Uso (desde la raíz del repositorio):
    python fuentes/datos_prep.py

Entrada : datos/datos_ini/ai4i2020.csv            (dataset original, sin modificar)
Salidas : datos/datos_limp/ai4i2020_limpio.csv     (dataset limpio, v1.1.0)
          datos/datos_limp/reporte_limpieza.json   (bitácora de cada regla aplicada)
          datos/datos_limp/diccionario_datos.json  (diccionario de datos)
          datos/MANIFEST.json                      (hashes SHA-256 para versionamiento)

Decisión de diseño: el escalado (StandardScaler) y la codificación one-hot NO se
aplican aquí, sino dentro del Pipeline de scikit-learn en train.py. Así el
escalador se ajusta solo con los datos de entrenamiento de cada fold y se evita
fuga de información (data leakage) hacia validación y prueba.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_INI = RAIZ / "datos" / "datos_ini" / "ai4i2020.csv"
DIR_LIMP = RAIZ / "datos" / "datos_limp"
RUTA_LIMP = DIR_LIMP / "ai4i2020_limpio.csv"

VERSION_DATOS = "1.1.0"

# ---------------------------------------------------------------------------
# 1. Estandarización de nombres
# ---------------------------------------------------------------------------
MAPA_COLUMNAS = {
    "UDI": "udi",
    "Product ID": "id_producto",
    "Type": "tipo",
    "Air temperature [K]": "temp_aire_k",
    "Process temperature [K]": "temp_proceso_k",
    "Rotational speed [rpm]": "vel_rotacion_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "desgaste_min",
    "Machine failure": "falla",
    "TWF": "f_desgaste_herr",
    "HDF": "f_disipacion_calor",
    "PWF": "f_potencia",
    "OSF": "f_sobreesfuerzo",
    "RNF": "f_aleatoria",
}
MODOS_FALLA = ["f_desgaste_herr", "f_disipacion_calor", "f_potencia",
               "f_sobreesfuerzo", "f_aleatoria"]
NUMERICAS_BASE = ["temp_aire_k", "temp_proceso_k", "vel_rotacion_rpm",
                  "torque_nm", "desgaste_min"]
TIPOS_VALIDOS = ["L", "M", "H"]

# Reglas de negocio: rangos físicamente plausibles para la línea (en unidades SI).
# Se definen con margen respecto a la especificación del proceso simulado.
RANGOS_FISICOS = {
    "temp_aire_k": (290.0, 310.0),
    "temp_proceso_k": (300.0, 320.0),
    "vel_rotacion_rpm": (1000.0, 3000.0),
    "torque_nm": (0.0, 100.0),
    "desgaste_min": (0.0, 260.0),
}

# Variables que se usan para entrenar (todo lo demás es identificador o fuga).
CARACTERISTICAS_NUM = NUMERICAS_BASE + ["delta_temp_k", "potencia_w", "desgaste_x_torque"]
CARACTERISTICAS_CAT = ["tipo"]
OBJETIVO = "falla"


def cargar_crudo(ruta: Path = RUTA_INI) -> pd.DataFrame:
    """Carga el CSV original (maneja el BOM UTF-8 que trae el archivo fuente)."""
    return pd.read_csv(ruta, encoding="utf-8-sig")


def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra a snake_case en español y quita espacios residuales."""
    df = df.rename(columns=lambda c: c.strip()).rename(columns=MAPA_COLUMNAS)
    faltantes = set(MAPA_COLUMNAS.values()) - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas esperadas no encontradas: {faltantes}")
    return df


def tipificar(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza tipos de datos explícitos (numéricos, categóricos y binarios)."""
    df = df.copy()
    for c in NUMERICAS_BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
    df.loc[~df["tipo"].isin(TIPOS_VALIDOS), "tipo"] = np.nan
    for c in [OBJETIVO] + MODOS_FALLA:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["udi"] = df["udi"].astype("int64")
    df["id_producto"] = df["id_producto"].astype(str).str.strip()
    return df


def imputar_nulos(df: pd.DataFrame, bitacora: dict) -> pd.DataFrame:
    """Política de nulos:
    - Objetivo nulo  -> se elimina el registro (no se imputa una etiqueta).
    - Numéricas      -> mediana por tipo de producto (robusta a atípicos).
    - Categórica     -> moda.
    """
    df = df.copy()
    nulos_antes = df.isna().sum()
    bitacora["nulos_por_columna"] = {k: int(v) for k, v in nulos_antes.items() if v > 0}

    n0 = len(df)
    df = df[df[OBJETIVO].notna()]
    bitacora["eliminados_objetivo_nulo"] = n0 - len(df)

    if df["tipo"].isna().any():
        df["tipo"] = df["tipo"].fillna(df["tipo"].mode().iat[0])
    for c in NUMERICAS_BASE:
        if df[c].isna().any():
            df[c] = df.groupby("tipo")[c].transform(lambda s: s.fillna(s.median()))
    bitacora["nulos_restantes"] = int(df.isna().sum().sum())
    return df


def eliminar_duplicados(df: pd.DataFrame, bitacora: dict) -> pd.DataFrame:
    """Criterios: (1) filas idénticas; (2) mismo UDI o mismo ID de producto
    (cada pieza es única en la línea); se conserva la primera ocurrencia."""
    n0 = len(df)
    df = df.drop_duplicates()
    d_exactos = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset=["udi"]).drop_duplicates(subset=["id_producto"])
    bitacora["duplicados_exactos"] = d_exactos
    bitacora["duplicados_por_id"] = n1 - len(df)
    return df


def validar_reglas_negocio(df: pd.DataFrame, bitacora: dict) -> pd.DataFrame:
    """Reglas de negocio:
    R1. Cada variable de sensor debe estar dentro de su rango físico.
    R2. La temperatura de proceso debe ser mayor que la del aire (el proceso calienta).
    R3. Consistencia de etiqueta: 'falla=1' debe tener al menos un modo de falla
        identificado y un modo de falla identificado implica 'falla=1'.
        Los registros que violan esto son ambiguos para supervisión y se excluyen.
    """
    reglas = {}
    mascara_ok = pd.Series(True, index=df.index)
    for c, (lo, hi) in RANGOS_FISICOS.items():
        fuera = ~df[c].between(lo, hi)
        reglas[f"R1_fuera_rango_{c}"] = int(fuera.sum())
        mascara_ok &= ~fuera

    r2 = df["temp_proceso_k"] <= df["temp_aire_k"]
    reglas["R2_proceso_no_mayor_aire"] = int(r2.sum())
    mascara_ok &= ~r2

    n_modos = df[MODOS_FALLA].sum(axis=1)
    falla_sin_modo = (df[OBJETIVO] == 1) & (n_modos == 0)
    modo_sin_falla = (df[OBJETIVO] == 0) & (n_modos > 0)
    reglas["R3_falla_sin_modo"] = int(falla_sin_modo.sum())
    reglas["R3_modo_sin_falla"] = int(modo_sin_falla.sum())
    mascara_ok &= ~(falla_sin_modo | modo_sin_falla)

    bitacora["reglas_negocio"] = reglas
    bitacora["eliminados_reglas_negocio"] = int((~mascara_ok).sum())
    return df[mascara_ok].copy()


def ingenieria_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas con significado físico (documentado en el reporte):
    - delta_temp_k       = T_proceso - T_aire       (capacidad de disipar calor)
    - potencia_w         = torque * ω, ω = rpm·2π/60 (potencia mecánica)
    - desgaste_x_torque  = desgaste * torque         (indicador de sobreesfuerzo)
    """
    df = df.copy()
    df["delta_temp_k"] = (df["temp_proceso_k"] - df["temp_aire_k"]).round(3)
    df["potencia_w"] = (df["torque_nm"] * df["vel_rotacion_rpm"] * 2 * np.pi / 60).round(2)
    df["desgaste_x_torque"] = (df["desgaste_min"] * df["torque_nm"]).round(2)
    return df


def marcar_atipicos(df: pd.DataFrame, bitacora: dict) -> pd.DataFrame:
    """Marca atípicos por IQR (1.5) SIN eliminarlos: en este dominio los valores
    extremos de torque/velocidad suelen ser justamente la señal de falla."""
    df = df.copy()
    resumen = {}
    marca = pd.Series(False, index=df.index)
    for c in ["vel_rotacion_rpm", "torque_nm"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        m = ~df[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        resumen[c] = {"n": int(m.sum()),
                      "tasa_falla_atipicos": round(float(df.loc[m, OBJETIVO].mean()), 4)}
        marca |= m
    df["atipico_iqr"] = marca.astype(int)
    bitacora["atipicos_iqr_conservados"] = resumen
    return df


def etiquetar_modo(df: pd.DataFrame) -> pd.DataFrame:
    """Columna descriptiva del modo de falla (solo para análisis, NO para entrenar)."""
    df = df.copy()
    nombres = {"f_desgaste_herr": "TWF", "f_disipacion_calor": "HDF",
               "f_potencia": "PWF", "f_sobreesfuerzo": "OSF", "f_aleatoria": "RNF"}

    def _modo(fila):
        activos = [nombres[m] for m in MODOS_FALLA if fila[m] == 1]
        return "+".join(activos) if activos else "sin_falla"

    df["modo_falla"] = df.apply(_modo, axis=1)
    return df


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


DICCIONARIO = {
    "udi": "Identificador único del registro (1–10000). Solo trazabilidad, no se entrena.",
    "id_producto": "ID de la pieza: letra de calidad + número de serie. No se entrena.",
    "tipo": "Calidad de la pieza: L (baja, 60%), M (media, 30%), H (alta, 10%). Categórica.",
    "temp_aire_k": "Temperatura del aire [K]. Paseo aleatorio normalizado, σ=2 K alrededor de 300 K.",
    "temp_proceso_k": "Temperatura de proceso [K] = temp. aire + 10 K + ruido σ=1 K.",
    "vel_rotacion_rpm": "Velocidad de rotación del husillo [rpm], derivada de una potencia de 2860 W con ruido.",
    "torque_nm": "Par [N·m], normal con media 40 N·m y σ=10, sin valores negativos.",
    "desgaste_min": "Minutos de uso de la herramienta; H/M/L agregan 5/3/2 min por ciclo.",
    "falla": "OBJETIVO. 1 si la máquina falló por cualquiera de los modos (binaria).",
    "f_desgaste_herr": "Modo TWF: reemplazo/falla por desgaste de herramienta (200–240 min). Fuga: no se entrena.",
    "f_disipacion_calor": "Modo HDF: ΔT < 8.6 K y rpm < 1380. Fuga: no se entrena.",
    "f_potencia": "Modo PWF: potencia < 3500 W o > 9000 W. Fuga: no se entrena.",
    "f_sobreesfuerzo": "Modo OSF: desgaste×torque > 11000 (L), 12000 (M), 13000 (H). Fuga: no se entrena.",
    "f_aleatoria": "Modo RNF: falla aleatoria con prob. 0.1%. Fuga: no se entrena.",
    "delta_temp_k": "DERIVADA. temp_proceso_k − temp_aire_k [K].",
    "potencia_w": "DERIVADA. torque_nm × rpm × 2π/60 [W].",
    "desgaste_x_torque": "DERIVADA. desgaste_min × torque_nm [min·N·m].",
    "atipico_iqr": "Marca 1 si rpm o torque están fuera de 1.5·IQR (informativa, no se entrena).",
    "modo_falla": "Etiqueta textual del modo de falla para análisis exploratorio.",
}


def pipeline_limpieza(ruta_ini: Path = RUTA_INI, ruta_out: Path = RUTA_LIMP,
                      escribir: bool = True) -> tuple[pd.DataFrame, dict]:
    """Ejecuta todas las etapas en orden y devuelve (df_limpio, bitácora)."""
    bit = {"version_datos": VERSION_DATOS,
           "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    df = cargar_crudo(ruta_ini)
    bit["filas_iniciales"] = len(df)
    df = estandarizar_columnas(df)
    df = tipificar(df)
    df = imputar_nulos(df, bit)
    df = eliminar_duplicados(df, bit)
    df = validar_reglas_negocio(df, bit)
    df = ingenieria_caracteristicas(df)
    df = marcar_atipicos(df, bit)
    df = etiquetar_modo(df)
    df[OBJETIVO] = df[OBJETIVO].astype(int)
    for m in MODOS_FALLA:
        df[m] = df[m].astype(int)
    df = df.sort_values("udi").reset_index(drop=True)
    bit["filas_finales"] = len(df)
    bit["tasa_falla_final"] = round(float(df[OBJETIVO].mean()), 4)
    bit["distribucion_tipo"] = df["tipo"].value_counts().to_dict()

    if escribir:
        ruta_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_out, index=False)
        (ruta_out.parent / "reporte_limpieza.json").write_text(
            json.dumps(bit, indent=2, ensure_ascii=False), encoding="utf-8")
        (ruta_out.parent / "diccionario_datos.json").write_text(
            json.dumps(DICCIONARIO, indent=2, ensure_ascii=False), encoding="utf-8")
        manifiesto = {
            "version_datos": VERSION_DATOS,
            "fuente": "AI4I 2020 Predictive Maintenance Dataset — UCI ML Repository (id 601)",
            "licencia": "CC BY 4.0",
            "archivos": {
                str(ruta_ini.relative_to(RAIZ)): {"sha256": sha256(ruta_ini), "filas": bit["filas_iniciales"]},
                str(ruta_out.relative_to(RAIZ)): {"sha256": sha256(ruta_out), "filas": bit["filas_finales"]},
            },
        }
        (RAIZ / "datos" / "MANIFEST.json").write_text(
            json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return df, bit


def cargar_limpio(ruta: Path = RUTA_LIMP) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df["tipo"] = pd.Categorical(df["tipo"], categories=TIPOS_VALIDOS)
    return df


def separar_xy(df: pd.DataFrame):
    """Devuelve X (solo variables permitidas) e y; excluye ids y modos de falla (fuga)."""
    return df[CARACTERISTICAS_NUM + CARACTERISTICAS_CAT].copy(), df[OBJETIVO].astype(int)


if __name__ == "__main__":
    _, b = pipeline_limpieza()
    print(json.dumps(b, indent=2, ensure_ascii=False))
