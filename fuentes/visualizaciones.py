"""
visualizaciones.py — Gráficas de EDA y evaluación (compartidas por el notebook y train.py).
Paleta categórica fija (slot 1 azul, 2 naranja, 3 aqua) validada para daltonismo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                             roc_curve, auc)

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLORES_MODELO = {"regresion_logistica": AZUL, "random_forest": NARANJA, "xgboost": AQUA}
NOMBRES_MODELO = {"regresion_logistica": "Regresión logística",
                  "random_forest": "Random Forest", "xgboost": "XGBoost"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "text.color": TINTA,
})

VARS = {"temp_aire_k": "Temp. aire [K]", "temp_proceso_k": "Temp. proceso [K]",
        "vel_rotacion_rpm": "Velocidad [rpm]", "torque_nm": "Torque [N·m]",
        "desgaste_min": "Desgaste [min]", "delta_temp_k": "ΔT [K]",
        "potencia_w": "Potencia [W]", "desgaste_x_torque": "Desgaste × torque"}


def _guardar(fig, ruta: Path | None):
    if ruta is not None:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta)
    return fig


def balance_clases(df, ruta=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cnt = df["falla"].value_counts().sort_index()
    b = axes[0].bar(["Sin falla (0)", "Falla (1)"], cnt.values, color=[AZUL, NARANJA], width=0.55)
    for r, v in zip(b, cnt.values):
        axes[0].text(r.get_x() + r.get_width() / 2, v, f"{v:,}\n({v/cnt.sum():.1%})",
                     ha="center", va="bottom", fontsize=9, color=TINTA)
    axes[0].set_title("Desbalance de la variable objetivo")
    axes[0].set_ylim(0, cnt.max() * 1.18)
    modos = df.loc[df["falla"] == 1, "modo_falla"].value_counts()
    axes[1].barh(modos.index[::-1], modos.values[::-1], color=NARANJA, height=0.55)
    for i, v in enumerate(modos.values[::-1]):
        axes[1].text(v + 1, i, str(v), va="center", fontsize=9)
    axes[1].set_title("Fallas por modo (puede haber más de uno)")
    axes[1].set_xlabel("Registros")
    fig.tight_layout()
    return _guardar(fig, ruta)


def distribuciones(df, ruta=None):
    cols = list(VARS)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), cols):
        for y, col, lab in [(0, AZUL, "Sin falla"), (1, NARANJA, "Falla")]:
            ax.hist(df.loc[df["falla"] == y, c], bins=40, density=True, alpha=0.55,
                    color=col, label=lab, edgecolor="white", linewidth=0.3)
        ax.set_title(VARS[c])
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Distribución por clase (densidad normalizada)", fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, ruta)


def boxplots_atipicos(df, ruta=None):
    cols = ["vel_rotacion_rpm", "torque_nm", "desgaste_min", "potencia_w"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, c in zip(axes, cols):
        datos = [df.loc[df["falla"] == 0, c], df.loc[df["falla"] == 1, c]]
        bp = ax.boxplot(datos, tick_labels=["Sin falla", "Falla"], patch_artist=True, widths=0.5,
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                                        markeredgecolor=TINTA2))
        for p, col in zip(bp["boxes"], [AZUL, NARANJA]):
            p.set_facecolor(col); p.set_alpha(0.6); p.set_edgecolor(TINTA2)
        for m in bp["medians"]:
            m.set_color(TINTA)
        ax.set_title(VARS[c])
    fig.suptitle("Atípicos (IQR 1.5) — se concentran en la clase falla", fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, ruta)


def correlaciones(df, ruta=None):
    cols = list(VARS) + ["falla"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    etiquetas = [VARS.get(c, "Falla") for c in cols]
    ax.set_xticks(range(len(cols)), etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), etiquetas)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else TINTA)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    return _guardar(fig, ruta)


def mapas_falla(df, ruta=None):
    """Dispersión de las variables físicas clave: los modos de falla aparecen como regiones."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ok, f = df[df["falla"] == 0], df[df["falla"] == 1]
    axes[0].scatter(ok["vel_rotacion_rpm"], ok["torque_nm"], s=4, color=AZUL, alpha=0.25, label="Sin falla")
    axes[0].scatter(f["vel_rotacion_rpm"], f["torque_nm"], s=14, color=NARANJA, alpha=0.9,
                    edgecolor="white", linewidth=0.4, label="Falla")
    rpm = np.linspace(1150, 2900, 200)
    for p in (3500, 9000):
        axes[0].plot(rpm, p / (rpm * 2 * np.pi / 60), ls="--", color=TINTA2, lw=1)
        axes[0].text(2850, p / (2850 * 2 * np.pi / 60) + 1.5, f"{p} W", fontsize=8, color=TINTA2, ha="right")
    axes[0].set_xlabel("Velocidad [rpm]"); axes[0].set_ylabel("Torque [N·m]")
    axes[0].set_title("Velocidad vs torque (isolíneas de potencia)")
    axes[0].legend(loc="upper right", markerscale=2)
    axes[1].scatter(ok["desgaste_min"], ok["desgaste_x_torque"], s=4, color=AZUL, alpha=0.25)
    axes[1].scatter(f["desgaste_min"], f["desgaste_x_torque"], s=14, color=NARANJA, alpha=0.9,
                    edgecolor="white", linewidth=0.4)
    axes[1].axhline(11000, ls="--", color=TINTA2, lw=1)
    axes[1].text(5, 11300, "umbral sobreesfuerzo (L) = 11 000", fontsize=8, color=TINTA2)
    axes[1].axvspan(200, 240, color=GRILLA, alpha=0.6, lw=0, zorder=0)
    axes[1].text(203, 500, "zona TWF\n200–240 min", fontsize=8, color=TINTA2)
    axes[1].set_xlabel("Desgaste [min]"); axes[1].set_ylabel("Desgaste × torque")
    axes[1].set_title("Desgaste de herramienta y sobreesfuerzo")
    fig.tight_layout()
    return _guardar(fig, ruta)


def tasa_por_tipo(df, ruta=None):
    t = df.groupby("tipo", observed=True)["falla"].agg(["mean", "size"]).reindex(["L", "M", "H"])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    b = ax.bar(t.index, t["mean"] * 100, color=AZUL, width=0.5)
    for r, (m, n) in zip(b, t.values):
        ax.text(r.get_x() + r.get_width() / 2, m * 100, f"{m:.2%}\nn={int(n):,}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Tasa de falla [%]"); ax.set_ylim(0, t["mean"].max() * 130)
    ax.set_title("Tasa de falla por calidad de pieza")
    fig.tight_layout()
    return _guardar(fig, ruta)


# --------------------------- Evaluación de modelos ---------------------------

def matriz_confusion(y, yhat, titulo, ruta=None):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(y, yhat, display_labels=["Sin falla", "Falla"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.grid(False); ax.set_title(titulo)
    fig.tight_layout()
    return _guardar(fig, ruta)


def curvas_roc_pr(resultados: dict, y, ruta=None):
    """resultados = {nombre_modelo: probas_test}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for nom, p in resultados.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        col = COLORES_MODELO[nom]
        axes[0].plot(fpr, tpr, lw=2, color=col, label=f"{NOMBRES_MODELO[nom]} (AUC={auc(fpr, tpr):.3f})")
        axes[1].plot(rc, pr, lw=2, color=col, label=f"{NOMBRES_MODELO[nom]} (AP={auc(rc, pr):.3f})")
    axes[0].plot([0, 1], [0, 1], ls=":", color=TINTA2, lw=1)
    axes[0].set_xlabel("Tasa de falsos positivos"); axes[0].set_ylabel("Sensibilidad (recall)")
    axes[0].set_title("Curva ROC — conjunto de prueba")
    base = float(np.mean(y))
    axes[1].axhline(base, ls=":", color=TINTA2, lw=1)
    axes[1].text(0.02, base + 0.02, f"línea base = {base:.3f}", fontsize=8, color=TINTA2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisión")
    axes[1].set_title("Curva precisión-recall — conjunto de prueba")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8.5)
    fig.tight_layout()
    return _guardar(fig, ruta)


def barras_metricas(tabla: pd.DataFrame, ruta=None):
    """tabla: índice = modelo, columnas = métricas (prueba)."""
    metricas = list(tabla.columns)
    x = np.arange(len(metricas)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, nom in enumerate(tabla.index):
        vals = tabla.loc[nom].values
        b = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLORES_MODELO[nom], label=NOMBRES_MODELO[nom])
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, metricas); ax.set_ylim(0, 1.1)
    ax.set_title("Métricas en el conjunto de prueba (umbral 0.5)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return _guardar(fig, ruta)


def boxplot_cv(scores: dict, metrica: str, ruta=None):
    """scores = {modelo: array de puntajes por fold (CV repetida)}"""
    noms = list(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([scores[n] for n in noms], tick_labels=[NOMBRES_MODELO[n] for n in noms],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TINTA, markersize=5))
    for p, n in zip(bp["boxes"], noms):
        p.set_facecolor(COLORES_MODELO[n]); p.set_alpha(0.7); p.set_edgecolor(TINTA2)
    for m in bp["medians"]:
        m.set_color(TINTA)
    rng = np.random.default_rng(0)
    for i, n in enumerate(noms, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(scores[n])), scores[n], s=9, color=TINTA2, alpha=0.5, zorder=3)
    ax.set_ylabel(metrica)
    ax.set_title(f"{metrica} en validación cruzada repetida (5×10 folds)")
    fig.tight_layout()
    return _guardar(fig, ruta)


def importancias(nombres, valores, titulo, color, ruta=None, top=11):
    s = pd.Series(valores, index=nombres).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index, s.values, color=color, height=0.6)
    ax.set_title(titulo)
    fig.tight_layout()
    return _guardar(fig, ruta)
