# Actividad 5 · Gestión de datos, entrenamiento y registro con MLflow

**Proyecto:** VisionQ — módulo predictivo de proceso · **Autor:** Rene

VisionQ inspecciona visualmente las piezas terminadas. Este módulo predice la **falla de la máquina** a partir de variables de proceso
(temperaturas, velocidad, torque, desgaste de herramienta) para intervenir antes de que la línea produzca defectos.

| | |
|---|---|
| Dataset | AI4I 2020 Predictive Maintenance (UCI 601, CC BY 4.0) |
| Tarea | Clasificación binaria desbalanceada (3.3% de fallas) |
| Modelos | Regresión logística · Random Forest · XGBoost |
| Ajuste | GridSearchCV + StratifiedKFold(5), `refit="f1"` |
| Registro | MLflow (`sqlite:///mlflow.db`), Model Registry con alias `champion` |
| Campeón | **Random Forest** — F1 prueba 0.906 · recall 0.879 · ROC-AUC 0.991 |

## Estructura
```
Actividad5/
├── datos/datos_ini/ai4i2020.csv          # original (datos-v1.0.0)
├── datos/datos_limp/ai4i2020_limpio.csv  # limpio (datos-v1.1.0) + diccionario + bitácora
├── datos/MANIFEST.json                   # SHA-256 de cada versión
├── fuentes/entrena.ipynb                 # pipeline completo (este cuaderno)
├── fuentes/datos_prep.py                 # limpieza y estandarización
├── fuentes/train.py                      # entrenamiento + MLflow (versión script)
├── fuentes/visualizaciones.py · capturas_mlflow.py · reubicar_mlflow.py
├── evidencias/                           # gráficas, tablas, capturas de MLflow
├── README.md · CHANGELOG.md · requirements.txt
```

## Reproducción
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute fuentes/entrena.ipynb     # todo el proceso
# o por scripts:
python fuentes/datos_prep.py && python fuentes/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Si recibe `mlflow.db`/`mlruns/` copiados de otra máquina, ejecute antes `python fuentes/reubicar_mlflow.py`.

## Decisiones de datos
- Nulos: etiqueta nula → eliminar; numéricas → mediana por tipo; categórica → moda.
- Duplicados por fila, `udi` e `id_producto`.
- Reglas: R1 rangos físicos, R2 T_proceso > T_aire, R3 consistencia de etiqueta (27 registros excluidos).
- Variables derivadas: ΔT, potencia (W), desgaste × torque. Atípicos marcados, no eliminados.
- Se excluyen identificadores y modos de falla (fuga). Escalado y one-hot dentro del Pipeline.

## Resultados
| Modelo | F1 CV (5×10) | F1 prueba [IC 95 %] | Recall | Precisión | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Regresión logística | 0.386 ± 0.077 | 0.349 [0.213, 0.471] | 0.227 | 0.750 | 0.947 | 0.529 |
| **Random Forest** | 0.892 ± 0.023 | 0.906 [0.849, 0.953] | 0.879 | 0.935 | 0.991 | 0.933 |
| XGBoost | 0.856 ± 0.034 | 0.844 [0.771, 0.905] | 0.818 | 0.871 | 0.992 | 0.921 |

Random Forest vs XGBoost: F1 p = 0.0219; ROC-AUC p = 0.855 (t corregida de Nadeau-Bengio, Holm).

## Publicar en GitHub
```bash
gh repo create Actividad5 --public --source=. --remote=origin --push
git push origin --tags
```

## Referencia
Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
