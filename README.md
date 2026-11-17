# SMC Full-Actuated Simulator — PlaDyPos (`usv5`)

Simulador en Python para el movimiento planar de 3 grados de libertad (Surge $u$, Sway $v$, Yaw $r$) del vehículo autónomo de superficie **PlaDyPos (`usv5`)** totalmente actuado con **4 propulsores en configuración X**.

## Configuración en archivos JSON

El modelo dinámico y las propiedades de los propulsores se cargan automáticamente desde dos archivos JSON:

1. **`model_params.json`**: Modelo dinámico de caja blanca de Fossen 3-DOF de 9 parámetros identificado para PlaDyPos (`m11`, `m22`, `m33`, `Xu`, `Xuu`, `Yv`, `Yvv`, `Nr`, `Nrr`).
2. **`thrusters_config.json`**: Curva logística generalizada asimétrica de **Richards** para los propulsores (+36.3827 N / -28.4393 N), parámetros PWM (1100–1900 µs) y las posiciones/ángulos físicos de los 4 propulsores en configuración X (`FR`, `FL`, `BR`, `BL`).

## Estructura de Archivos

| Archivo | Propósito |
| --- | --- |
| `model_params.json` | Parámetros del modelo de Fossen 3-DOF de 9 parámetros de PlaDyPos. |
| `thrusters_config.json` | Curva de empuje de Richards y posiciones/orientaciones de los 4 motores en X. |
| `run_simulation.py` | Punto de entrada principal: ejecuta la simulación, exporta CSV, gráficas PNG y animación GIF. |
| `usv_model.py` | Carga los JSON, construye la matriz de asignación $\Phi$, aplica la curva de Richards y simula la dinámica 3-DOF. |
| `controllers.py` | Trayectoria tipo 8, geometría de punto adelantado y controlador SMC 3-DOF totalmente actuado (`Tu`, `Tv`, `Tr`). |
| `cbf_filters.py` | Filtro de Control Barrier Functions (CBF) 3-DOF con barreras circular, elíptica y cono de colisión. |
| `reference_governor.py` | Gobernador de referencia en tiempo virtual ($\dot{\sigma} = \gamma$). |
| `obstacle_config.py` | Configuración de obstáculos circulares. |
| `disturbances.py` | Perturbaciones externas (viento, corriente direccional, constante, senoidal, combinada). |
| `plot_results.py` | Generación de gráficas de trayectoria, errores, velocidades 3-DOF, fuerzas `Tu/Tv/Tr`, propulsores en X, CBF y animación GIF. |
| `analysis_utils.py` | Utilidades para análisis de error y ciclo de trabajo de saturación. |
| `test_*.py` | Pruebas unitarias de carga JSON, conversión Richards/PWM, obstáculos e historial. |

## Ejecución

```bash
cd /home/brayan/ros2_ws/src/smc_full_actuated
python3 -m unittest discover -v
python3 run_simulation.py
```
