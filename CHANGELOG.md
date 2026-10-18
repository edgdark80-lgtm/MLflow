# CHANGELOG — Actividad 5 (VisionQ · módulo predictivo de proceso)

Versionamiento semántico independiente para **datos** (tags `datos-vX.Y.Z` + SHA-256 en `datos/MANIFEST.json`) y **modelos** (Model Registry de MLflow).

## [1.0.0] — 2026-10-18 · Reporte final
### Resultado
- **Campeón:** Random Forest (`visionq_falla_random_forest` v1, alias `@champion`) — `class_weight=None`, `max_depth=12`, `max_features=sqrt`, `min_samples_leaf=1`, `n_estimators=200`.
- **Prueba:** F1 0.906 · recall 0.879 · precisión 0.935 · ROC-AUC 0.991 · PR-AUC 0.933.
- **CV repetida 5×10 (F1):** Regresión logística 0.386 ± 0.077 · Random Forest 0.892 ± 0.023 · XGBoost 0.856 ± 0.034.
- Random Forest vs XGBoost: F1 p = 0.0219; ROC-AUC p = 0.855.

### Deuda técnica → acción MLOps
| Deuda | Acción |
|---|---|
| Umbral fijo 0.5 | Calibrar probabilidades y elegir umbral por costo dentro de la CV |
| Dataset sintético y estático | Validar en modo sombra con datos reales de la línea |
| Versionado Git + SHA-256 | Migrar a DVC con almacenamiento remoto |
| Sin monitoreo de deriva | Evidently/whylogs; reentrenar si PSI > 0.2 |
| Contrato de entrada implícito | Validación de esquema (pandera) y prueba de contrato en CI |
| Entrenamiento manual | GitHub Actions: datos_prep → train → gate de métricas → registrar → mover alias |
| MLflow local (SQLite) | Servidor MLflow con PostgreSQL y artefactos remotos |
| Fallas aleatorias (RNF) | Documentadas como límite del modelo |

## [datos-v1.1.0] — 2026-10-18 · Dataset limpio
- `datos/datos_limp/ai4i2020_limpio.csv` (9,973 filas) · SHA-256 `430f8eb7…7b79979`.
- Variables derivadas `delta_temp_k`, `potencia_w`, `desgaste_x_torque`; columnas `atipico_iqr`, `modo_falla`.
- Eliminados por R3: 9 (falla sin modo) + 18 (modo sin falla).

## [datos-v1.0.0] — 2026-10-18 · Dataset original
- `datos/datos_ini/ai4i2020.csv` (10 000 × 14), sin modificar · SHA-256 `dc6630cd…bc8a8e`.

## [modelos] — 2026-10-18
- `visionq_falla_regresion_logistica` — `C=10.0`, `class_weight=None`, `penalty=l2`
- `visionq_falla_random_forest` — `class_weight=None`, `max_depth=12`, `max_features=sqrt`, `min_samples_leaf=1`, `n_estimators=200`
- `visionq_falla_xgboost` — `learning_rate=0.05`, `max_depth=3`, `n_estimators=400`, `scale_pos_weight=1.0`, `subsample=0.8`
- 88 runs en `Actividad5_VisionQ_FallaMaquina`.
