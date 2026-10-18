"""
capturas_mlflow.py — Toma capturas automáticas del panel de MLflow (evidencia de registro).

Requisitos: pip install playwright && python -m playwright install chromium
Uso (desde la raíz, con el servidor levantado en otra terminal):
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    python fuentes/capturas_mlflow.py --url http://127.0.0.1:5001
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "evidencias" / "mlflow"


def main(url: str, exp_name: str):
    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    c = MlflowClient()
    exp = c.get_experiment_by_name(exp_name)
    runs = c.search_runs([exp.experiment_id], "tags.tipo_run = 'grid_search'")
    comp = c.search_runs([exp.experiment_id], "tags.tipo_run = 'comparacion'")
    padres = {r.data.tags["modelo"]: r.info.run_id for r in runs}
    eid = exp.experiment_id
    SALIDA.mkdir(parents=True, exist_ok=True)

    vistas = [
        ("01_experimento_runs.png", f"{url}/#/experiments/{eid}/runs", None),
        ("02_comparacion_runs_padre.png",
         f"{url}/#/compare-runs?runs={str(list(padres.values())).replace(chr(39), '%22')}&experiments=[%22{eid}%22]", None),
        ("02b_comparacion_metricas.png",
         f"{url}/#/compare-runs?runs={str(list(padres.values())).replace(chr(39), '%22')}&experiments=[%22{eid}%22]",
         "Metrics"),
        ("06_modelos_registrados.png", f"{url}/#/models", None),
        ("06b_modelo_campeon.png", f"{url}/#/models/visionq_falla_random_forest", None),
    ]
    for nom, rid in padres.items():
        vistas.append((f"03_run_{nom}_overview.png", f"{url}/#/experiments/{eid}/runs/{rid}", None))
        vistas.append((f"04_run_{nom}_metricas.png", f"{url}/#/experiments/{eid}/runs/{rid}/model-metrics", None))
        vistas.append((f"05_run_{nom}_artefactos.png", f"{url}/#/experiments/{eid}/runs/{rid}/artifacts/graficas/cm_{nom}.png", None))
    if comp:
        rid = comp[0].info.run_id
        vistas.append(("07_run_comparacion_estadistica.png", f"{url}/#/experiments/{eid}/runs/{rid}", None))
        vistas.append(("08_comparacion_artefactos.png",
                       f"{url}/#/experiments/{eid}/runs/{rid}/artifacts/graficas/boxplot_cv_f1.png", None))

    with sync_playwright() as p:
        nav = p.chromium.launch()
        pag = nav.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        cerrado = False
        for nombre, destino, ancla in vistas:
            pag.goto(destino, wait_until="load")
            time.sleep(6)
            if not cerrado:  # cierra el panel lateral del asistente para ganar espacio
                try:
                    pag.locator("button[aria-label*='lose']").last.click(timeout=2000)
                    cerrado = True
                    time.sleep(1)
                except Exception:
                    pass
            if ancla:
                try:
                    pag.get_by_text(ancla, exact=True).last.scroll_into_view_if_needed(timeout=3000)
                    time.sleep(1)
                except Exception:
                    pass
            pag.screenshot(path=str(SALIDA / nombre), full_page=False)
            print("✓", nombre)
        nav.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--experimento", default="Actividad5_VisionQ_FallaMaquina")
    a = ap.parse_args()
    main(a.url, a.experimento)
