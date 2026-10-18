"""
train.py — Entrenamiento, ajuste de hiperparámetros (Grid Search + CV estratificada)
y registro de experimentos en MLflow.

Proyecto: Actividad 5 · Módulo predictivo de proceso para VisionQ
Autor: Rene

Ejecutar desde la raíz del repositorio:
    python fuentes/datos_prep.py          # (si aún no existe el dataset limpio)
    python fuentes/train.py               # entrena y registra en ./mlflow.db
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Qué se registra en MLflow
  * 1 run padre por algoritmo (gs_<modelo>) con:
      - parámetros: espacio de búsqueda, mejores hiperparámetros, semilla, versión de datos
      - métricas: media ± desv. de CV (5 folds) y métricas en prueba
      - artefactos: cv_results.csv, matriz de confusión, importancias, modelo (+ firma)
      - dataset de entrada (mlflow.data) con su hash, y tags de trazabilidad (git, sha256)
  * 1 run hijo (anidado) por cada combinación del grid → trazabilidad completa del ajuste
  * 1 run "comparacion_estadistica" con CV repetida 5×10, Friedman, t corregida
    (Nadeau & Bengio), Wilcoxon, McNemar e IC bootstrap
  * Model Registry: un modelo registrado por algoritmo; alias "champion" al mejor.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402
import visualizaciones as vz  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 42
TAM_PRUEBA = 0.20
K_FOLDS = 5
METRICA_REFIT = "f1"
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"}


# --------------------------------------------------------------------------- utilidades
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sin_git"


def tipos_confiables(modelo) -> list[str]:
    """MLflow 3 serializa con skops y exige declarar tipos no estándar. Como el modelo
    lo entrenamos nosotros en este mismo proceso (origen confiable), se declaran
    exactamente los tipos que skops reporta para ESTE objeto, no una lista genérica."""
    import skops.io as sio
    return sio.get_untrusted_types(data=sio.dumps(modelo))


def preprocesador() -> ColumnTransformer:
    """Escalado + one-hot dentro del pipeline: se ajusta solo con el fold de entrenamiento."""
    return ColumnTransformer([
        ("num", StandardScaler(), dp.CARACTERISTICAS_NUM),
        ("cat", OneHotEncoder(categories=[dp.TIPOS_VALIDOS], handle_unknown="ignore",
                              sparse_output=False), dp.CARACTERISTICAS_CAT),
    ])


def definir_modelos(spw: float) -> dict:
    """Algoritmos y espacio de búsqueda (justificación técnica en el README/reporte)."""
    return {
        "regresion_logistica": {
            "estimador": LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEMILLA),
            "grid": {
                # C cubre 4 órdenes de magnitud: de regularización fuerte a casi nula.
                "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                # L1 permite selección de variables; L2 maneja la colinealidad (rpm–torque).
                "clf__penalty": ["l1", "l2"],
                # 'balanced' compensa el desbalance 30:1 sin sobremuestreo sintético.
                "clf__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimador": RandomForestClassifier(random_state=SEMILLA, n_jobs=1),
            "grid": {
                "clf__n_estimators": [200, 400],          # estabilidad vs. costo (M1 8 GB)
                "clf__max_depth": [None, 12],              # control de sobreajuste
                "clf__min_samples_leaf": [1, 3],           # suaviza hojas con pocos positivos
                "clf__max_features": ["sqrt", 0.5],        # decorrelación de árboles
                "clf__class_weight": [None, "balanced_subsample"],
            },
        },
        "xgboost": {
            "estimador": XGBClassifier(eval_metric="logloss", tree_method="hist",
                                       random_state=SEMILLA, n_jobs=1, verbosity=0),
            "grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 6],                  # interacciones de 2º–5º orden
                "clf__learning_rate": [0.05, 0.1],         # tasa vs. nº de árboles
                "clf__subsample": [0.8, 1.0],              # bagging estocástico
                "clf__scale_pos_weight": [1.0, round(spw, 1)],  # ratio neg/pos del train
            },
        },
    }


def metricas_prueba(y, yhat, proba) -> dict:
    return {
        "test_accuracy": accuracy_score(y, yhat),
        "test_precision": precision_score(y, yhat, zero_division=0),
        "test_recall": recall_score(y, yhat),
        "test_f1": f1_score(y, yhat),
        "test_roc_auc": roc_auc_score(y, proba),
        "test_average_precision": average_precision_score(y, proba),
    }


def nombres_caracteristicas(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def importancia_modelo(pipe: Pipeline) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    return np.abs(clf.coef_.ravel())


# ------------------------------------------------------------- pruebas estadísticas
def t_corregida(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """t de Student corregida por remuestreo (Nadeau & Bengio, 2003) para CV repetida:
    var_corr = (1/J + n_test/n_train)·s²  con J = nº total de folds."""
    d = a - b
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def holm(pvals: dict) -> dict:
    """Corrección de Holm–Bonferroni para comparaciones múltiples."""
    orden = sorted(pvals.items(), key=lambda kv: kv[1])
    m, ajust, acum = len(orden), {}, 0.0
    for i, (k, p) in enumerate(orden):
        acum = max(acum, min(1.0, (m - i) * p))
        ajust[k] = acum
    return ajust


def mcnemar_exacta(y, pa, pb) -> tuple[int, int, float]:
    """McNemar exacta (binomial) sobre los desacuerdos de dos clasificadores en prueba."""
    a_ok, b_ok = pa == y, pb == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return n01, n10, p


def bootstrap_ic(y, yhat, proba, n=2000, semilla=SEMILLA) -> dict:
    rng = np.random.default_rng(semilla)
    y, yhat, proba = np.asarray(y), np.asarray(yhat), np.asarray(proba)
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        f1s.append(f1_score(y[idx], yhat[idx]))
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"f1_ic95": q(f1s), "roc_auc_ic95": q(aucs)}


# ------------------------------------------------------------------------ principal
def main(args):
    t0 = time.time()
    np.random.seed(SEMILLA)
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(args.experimento)
    client = MlflowClient()
    client.set_experiment_tag(exp.experiment_id, "proyecto", "VisionQ")
    client.set_experiment_tag(exp.experiment_id, "mlflow.note.content",
                              "Actividad 5 — Predicción de falla de máquina (AI4I 2020) como módulo "
                              "de proceso complementario a la inspección visual VisionQ. "
                              "Comparación: Regresión logística vs Random Forest vs XGBoost.")

    df = dp.cargar_limpio()
    X, y = dp.separar_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TAM_PRUEBA, stratify=y,
                                              random_state=SEMILLA)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    manifiesto = json.loads((RAIZ / "datos" / "MANIFEST.json").read_text(encoding="utf-8"))
    sha_limpio = manifiesto["archivos"]["datos/datos_limp/ai4i2020_limpio.csv"]["sha256"]
    ds_train = mlflow.data.from_pandas(X_tr.assign(falla=y_tr.values), name="ai4i2020_limpio_train",
                                       source=str(dp.RUTA_LIMP.relative_to(RAIZ)), targets="falla")
    ds_test = mlflow.data.from_pandas(X_te.assign(falla=y_te.values), name="ai4i2020_limpio_test",
                                      source=str(dp.RUTA_LIMP.relative_to(RAIZ)), targets="falla")

    tags_comunes = {
        "proyecto": "VisionQ", "actividad": "5", "autor": "Rene",
        "dataset": "AI4I 2020 (UCI 601)", "version_datos": manifiesto["version_datos"],
        "sha256_datos_limpios": sha_limpio[:16], "git_commit": git_commit(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}  positivos train={int(y_tr.sum())}  spw={spw:.1f}")

    out = RAIZ / "evidencias"
    (out / "graficas").mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)
    mejores, probas, preds, resumen = {}, {}, {}, []

    for nombre, cfg in definir_modelos(spw).items():
        pipe = Pipeline([("prep", preprocesador()), ("clf", cfg["estimador"])])
        n_comb = int(np.prod([len(v) for v in cfg["grid"].values()]))
        print(f"\n=== {nombre}: {n_comb} combinaciones × {K_FOLDS} folds ===")
        with mlflow.start_run(run_name=f"gs_{nombre}") as padre:
            mlflow.set_tags({**tags_comunes, "modelo": nombre, "tipo_run": "grid_search"})
            mlflow.log_input(ds_train, context="training")
            mlflow.log_input(ds_test, context="testing")
            mlflow.log_params({"semilla": SEMILLA, "tam_prueba": TAM_PRUEBA, "k_folds": K_FOLDS,
                               "metrica_refit": METRICA_REFIT, "n_combinaciones": n_comb,
                               "espacio_busqueda": json.dumps(cfg["grid"])})

            gs = GridSearchCV(pipe, cfg["grid"], scoring=SCORING, refit=METRICA_REFIT, cv=cv,
                              n_jobs=args.n_jobs, return_train_score=True)
            t1 = time.time()
            gs.fit(X_tr, y_tr)
            dur = time.time() - t1

            # --- runs hijos: uno por combinación del grid
            res = pd.DataFrame(gs.cv_results_)
            for i, fila in res.iterrows():
                with mlflow.start_run(run_name=f"{nombre}_c{i:02d}", nested=True):
                    mlflow.set_tags({"modelo": nombre, "tipo_run": "candidato_grid",
                                     "rank_f1": int(fila["rank_test_f1"])})
                    mlflow.log_params({k.replace("clf__", ""): v for k, v in fila["params"].items()})
                    m = {}
                    for met in SCORING:
                        m[f"cv_{met}_mean"] = fila[f"mean_test_{met}"]
                        m[f"cv_{met}_std"] = fila[f"std_test_{met}"]
                    m["cv_f1_train_mean"] = fila["mean_train_f1"]
                    m["fit_time_mean_s"] = fila["mean_fit_time"]
                    mlflow.log_metrics(m)

            # --- run padre: mejor configuración
            b = gs.best_index_
            mlflow.log_params({f"best_{k.replace('clf__', '')}": v for k, v in gs.best_params_.items()})
            cvm = {f"cv_{met}_mean": res.loc[b, f"mean_test_{met}"] for met in SCORING}
            cvm.update({f"cv_{met}_std": res.loc[b, f"std_test_{met}"] for met in SCORING})
            cvm["cv_f1_train_mean"] = res.loc[b, "mean_train_f1"]
            cvm["brecha_sobreajuste_f1"] = res.loc[b, "mean_train_f1"] - res.loc[b, "mean_test_f1"]
            cvm["tiempo_grid_s"] = dur
            mlflow.log_metrics(cvm)

            mejor = gs.best_estimator_
            proba = mejor.predict_proba(X_te)[:, 1]
            yhat = (proba >= 0.5).astype(int)
            mt = metricas_prueba(y_te, yhat, proba)
            mlflow.log_metrics(mt)

            # artefactos
            ruta_cv = out / f"cv_results_{nombre}.csv"
            res.drop(columns=["params"]).to_csv(ruta_cv, index=False)
            mlflow.log_artifact(str(ruta_cv), "tablas")
            f_cm = vz.matriz_confusion(y_te, yhat, f"{vz.NOMBRES_MODELO[nombre]} — prueba",
                                       out / "graficas" / f"cm_{nombre}.png")
            mlflow.log_figure(f_cm, f"graficas/cm_{nombre}.png")
            f_imp = vz.importancias(nombres_caracteristicas(mejor), importancia_modelo(mejor),
                                    f"Importancia de variables — {vz.NOMBRES_MODELO[nombre]}",
                                    vz.COLORES_MODELO[nombre],
                                    out / "graficas" / f"importancia_{nombre}.png")
            mlflow.log_figure(f_imp, f"graficas/importancia_{nombre}.png")
            vz.plt.close("all")

            firma = infer_signature(X_tr, mejor.predict_proba(X_tr)[:, 1])
            info = mlflow.sklearn.log_model(
                sk_model=mejor, name="modelo", signature=firma,
                input_example=X_tr.head(5), registered_model_name=f"visionq_falla_{nombre}",
                skops_trusted_types=tipos_confiables(mejor))

            mejores[nombre], probas[nombre], preds[nombre] = mejor, proba, yhat
            resumen.append({"modelo": nombre, "run_id": padre.info.run_id,
                            "model_uri": info.model_uri, "best_params": gs.best_params_,
                            **{k: round(v, 4) for k, v in cvm.items()},
                            **{k: round(v, 4) for k, v in mt.items()}})
            print(f"  mejor: {gs.best_params_}")
            print(f"  CV F1={cvm['cv_f1_mean']:.3f}±{cvm['cv_f1_std']:.3f}  "
                  f"Test F1={mt['test_f1']:.3f}  ROC-AUC={mt['test_roc_auc']:.3f}  ({dur:.0f}s)")

    # ------------------------------------------------------ comparación estadística
    print("\n=== Comparación estadística (CV repetida 5×%d) ===" % args.reps)
    with mlflow.start_run(run_name="comparacion_estadistica"):
        mlflow.set_tags({**tags_comunes, "tipo_run": "comparacion"})
        mlflow.log_params({"esquema_cv": f"RepeatedStratifiedKFold(5x{args.reps})",
                           "pruebas": "Friedman, t corregida Nadeau-Bengio, Wilcoxon, McNemar, bootstrap",
                           "correccion_multiple": "Holm-Bonferroni", "alfa": 0.05})
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.reps, random_state=SEMILLA)
        puntajes = {}
        for nombre, est in mejores.items():
            r = cross_validate(clone(est), X_tr, y_tr, cv=rcv, n_jobs=args.n_jobs,
                               scoring={"f1": "f1", "roc_auc": "roc_auc", "recall": "recall",
                                        "precision": "precision"})
            puntajes[nombre] = {k.replace("test_", ""): v for k, v in r.items() if k.startswith("test_")}
            for met, v in puntajes[nombre].items():
                mlflow.log_metric(f"rcv_{nombre}_{met}_mean", float(np.mean(v)))
                mlflow.log_metric(f"rcv_{nombre}_{met}_std", float(np.std(v, ddof=1)))
            print(f"  {nombre}: F1={np.mean(puntajes[nombre]['f1']):.3f}±{np.std(puntajes[nombre]['f1'], ddof=1):.3f}")

        n_tr_fold = int(len(X_tr) * 4 / 5); n_te_fold = len(X_tr) - n_tr_fold
        noms = list(mejores)
        filas_est = []
        for met in ["f1", "roc_auc"]:
            fr = stats.friedmanchisquare(*[puntajes[n][met] for n in noms])
            mlflow.log_metric(f"friedman_{met}_p", float(fr.pvalue))
            filas_est.append({"metrica": met, "prueba": "Friedman (3 modelos)", "par": "todos",
                              "estadistico": round(float(fr.statistic), 3), "p": float(fr.pvalue)})
            pt, pw = {}, {}
            for a, bb in itertools.combinations(noms, 2):
                t, p = t_corregida(puntajes[a][met], puntajes[bb][met], n_tr_fold, n_te_fold)
                w = stats.wilcoxon(puntajes[a][met], puntajes[bb][met])
                pt[(a, bb)] = p; pw[(a, bb)] = float(w.pvalue)
                filas_est.append({"metrica": met, "prueba": "t corregida (N&B)", "par": f"{a} vs {bb}",
                                  "estadistico": round(t, 3), "p": p,
                                  "dif_media": round(float(np.mean(puntajes[a][met] - puntajes[bb][met])), 4)})
                filas_est.append({"metrica": met, "prueba": "Wilcoxon", "par": f"{a} vs {bb}",
                                  "estadistico": round(float(w.statistic), 3), "p": float(w.pvalue),
                                  "dif_media": round(float(np.mean(puntajes[a][met] - puntajes[bb][met])), 4)})
            for nombre_p, dic in [("t corregida (N&B)", holm(pt)), ("Wilcoxon", holm(pw))]:
                for (a, bb), padj in dic.items():
                    for f in filas_est:
                        if f["metrica"] == met and f["prueba"] == nombre_p and f["par"] == f"{a} vs {bb}":
                            f["p_holm"] = padj
        for a, bb in itertools.combinations(noms, 2):
            n01, n10, p = mcnemar_exacta(y_te.values, preds[a], preds[bb])
            filas_est.append({"metrica": "aciertos test", "prueba": "McNemar exacta", "par": f"{a} vs {bb}",
                              "estadistico": f"{n01}/{n10}", "p": p})
        tabla_est = pd.DataFrame(filas_est)
        tabla_est["significativo_0.05"] = tabla_est.apply(
            lambda r: (r.get("p_holm") if pd.notna(r.get("p_holm", np.nan)) else r["p"]) < 0.05, axis=1)
        ruta_est = out / "comparacion_estadistica.csv"
        tabla_est.to_csv(ruta_est, index=False)
        mlflow.log_artifact(str(ruta_est), "tablas")

        ics = {n: bootstrap_ic(y_te, preds[n], probas[n]) for n in noms}
        for n, d in ics.items():
            mlflow.log_metrics({f"boot_{n}_f1_lo": d["f1_ic95"][0], f"boot_{n}_f1_hi": d["f1_ic95"][1],
                                f"boot_{n}_auc_lo": d["roc_auc_ic95"][0], f"boot_{n}_auc_hi": d["roc_auc_ic95"][1]})

        # figuras comparativas
        tabla_test = pd.DataFrame({r["modelo"]: {"Accuracy": r["test_accuracy"], "Precision": r["test_precision"],
                                                 "Recall": r["test_recall"], "F1": r["test_f1"],
                                                 "ROC-AUC": r["test_roc_auc"], "PR-AUC (AP)": r["test_average_precision"]}
                                   for r in resumen}).T
        figs = {
            "comparacion_metricas_test.png": vz.barras_metricas(tabla_test),
            "curvas_roc_pr.png": vz.curvas_roc_pr(probas, y_te),
            "boxplot_cv_f1.png": vz.boxplot_cv({n: puntajes[n]["f1"] for n in noms}, "F1"),
            "boxplot_cv_rocauc.png": vz.boxplot_cv({n: puntajes[n]["roc_auc"] for n in noms}, "ROC-AUC"),
        }
        for nom_f, fig in figs.items():
            fig.savefig(out / "graficas" / nom_f)
            mlflow.log_figure(fig, f"graficas/{nom_f}")
        vz.plt.close("all")

        # campeón: mayor F1 medio en CV repetida (no en prueba, para no sesgar la selección)
        campeon = max(noms, key=lambda n: np.mean(puntajes[n]["f1"]))
        mlflow.set_tag("campeon", campeon)
        try:
            versiones = client.search_model_versions(f"name='visionq_falla_{campeon}'")
            ultima = max(versiones, key=lambda v: int(v.version))
            client.set_registered_model_alias(f"visionq_falla_{campeon}", "champion", ultima.version)
            client.set_model_version_tag(f"visionq_falla_{campeon}", ultima.version, "validacion", "aprobado_cv_repetida")
        except Exception as e:  # el registro es opcional en tracking por archivos
            print("  aviso registry:", e)

    # resumen final en disco
    resumen_out = {"campeon": campeon, "duracion_total_s": round(time.time() - t0, 1),
                   "spw_train": round(spw, 2), "modelos": resumen,
                   "rcv": {n: {m: [float(np.mean(v)), float(np.std(v, ddof=1))]
                               for m, v in puntajes[n].items()} for n in noms},
                   "bootstrap_ic95": ics}
    (out / "resumen_resultados.json").write_text(
        json.dumps(resumen_out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tabla_test.round(4).to_csv(out / "metricas_test.csv")
    print(f"\nCampeón: {campeon}   ·   tiempo total {time.time() - t0:.0f}s")
    print(tabla_test.round(3).to_string())
    print(tabla_est.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", default=f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    ap.add_argument("--experimento", default="Actividad5_VisionQ_FallaMaquina")
    ap.add_argument("--n-jobs", type=int, default=-1, help="núcleos para GridSearchCV (-1 = todos)")
    ap.add_argument("--reps", type=int, default=10, help="repeticiones de la CV 5-fold para la comparación")
    main(ap.parse_args())
