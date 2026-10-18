"""
reubicar_mlflow.py — Hace portable el almacén de MLflow entregado (mlflow.db + mlruns/).

MLflow guarda rutas ABSOLUTAS a los artefactos. Si el repositorio se copia a otra
máquina, los parámetros y métricas se ven, pero los artefactos y modelos no se
encuentran. Este script reemplaza la ruta original por la ruta actual del repo.

Uso (desde la raíz del repositorio, con mlflow.db y mlruns/ presentes):
    python fuentes/reubicar_mlflow.py
"""
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DB = RAIZ / "mlflow.db"
COLUMNAS = [("experiments", "artifact_location"), ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"), ("model_versions", "storage_location")]


def main():
    con = sqlite3.connect(DB)
    fila = con.execute("SELECT artifact_location FROM experiments WHERE artifact_location LIKE '%/mlruns/%' LIMIT 1").fetchone()
    if not fila:
        print("No hay rutas que reubicar."); return
    viejo = fila[0].split("/mlruns/")[0]
    nuevo = (RAIZ).as_posix()
    if viejo == nuevo:
        print("El almacén ya apunta a", nuevo); return
    total = 0
    for tabla, col in COLUMNAS:
        try:
            cur = con.execute(f"UPDATE {tabla} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
                              (viejo, nuevo, f"%{viejo}%"))
            total += cur.rowcount
        except sqlite3.OperationalError:
            pass  # columna inexistente en esta versión de MLflow
    con.commit(); con.close()
    print(f"{total} rutas reubicadas: {viejo} -> {nuevo}")


if __name__ == "__main__":
    main()
