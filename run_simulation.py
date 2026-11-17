from pathlib import Path

import numpy as np

from cbf_filters import (
    CBF_ALPHA_1,
    CBF_ALPHA_2,
    CBF_BARRIER_COLLISION_CONE,
    CBF_BARRIER_ELLIPSE,
    CBF_BARRIER_TYPE,
    CBF_ELLIPSE_ANGLE_DEG,
    CBF_ELLIPSE_MAJOR_SCALE,
    CBF_ELLIPSE_MINOR_SCALE,
    CBF_MARGIN,
    cbf_filtered_controller,
    cbf_history,
)
from controllers import (
    FIGURE_EIGHT_LAPS,
    FIGURE_EIGHT_PERIOD,
    TRACKING_LOOKAHEAD,
    TRAJECTORY_CLOVER,
    TRAJECTORY_FIGURE_EIGHT,
    TRAJECTORY_SPIRAL,
    figure_eight_initial_state,
    figure_eight_lookahead_reference,
    figure_eight_reference,
    lookahead_point_from_state,
    set_active_trajectory,
    smc_figure_eight,
    trajectory_duration,
)
from disturbances import (
    INERTIAL_DISTURBANCE_CASES,
    combined_disturbance,
    constant_disturbance,
    directional_force_disturbance,
    no_disturbance,
    pure_sinusoidal_disturbance,
    sinusoidal_disturbance,
    wind_like_disturbance,
)
from obstacle_config import configured_obstacles
from plot_results import plot_results, save_simulation_animation
from reference_governor import (
    GOVERNOR_FILTER_TAU_MAX,
    GOVERNOR_FILTER_TAU_MIN,
    GOVERNOR_GAMMA_MIN,
    GOVERNOR_H_SLOW,
    GOVERNOR_H_STOP,
    governor_history,
    virtual_time_governor,
)
from usv_model import (
    MODEL_JSON_PATH,
    PARAMS,
    THRUSTERS_JSON_PATH,
    TR_LIMITS,
    TU_LIMITS,
    TV_LIMITS,
    simulate,
    thruster_forces_to_pwm_us,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "figure_eight_simulation.csv"
FIGURES_DIR = DATA_DIR / "figures"
ANIMATION_PATH = FIGURES_DIR / "Trayectory_animation.gif"
DISTURBANCES_FIGURES_DIR = DATA_DIR / "figures_disturbances"
OBSTACLES_FIGURES_DIR = DATA_DIR / "figures_obstacles"
OBSTACLES_DISTURBANCES_FIGURES_DIR = DATA_DIR / "figures_obstacles_disturbances"

DT = 0.02
SHOW_PLOTS = False
SAVE_ANIMATION = True
ANIMATION_FPS = 25
ANIMATION_DURATION = 20.0

# ============================================================
# Banderas principales de ejecucion en main():
#   - TRAJECTORY_TYPE      : Selecciona la trayectoria a ejecutar:
#                            "figure_eight" (8 de Lemniscata),
#                            "spiral"       (Espiral de Arquimedes),
#                            "clover"       (Trebol de 3 petalos / Rosa polar),
#                            "all"          (Ejecuta las 3 trayectorias en sus carpetas)
#   - INCLUDE_DISTURBANCES : Corre las 4 simulaciones con corrientes de agua
#   - INCLUDE_OBSTACLES    : Corre la simulacion con obstaculos (CBF + Governor)
#                            (Si ambas son True, tambien corre obstaculos + corrientes)
# ============================================================
TRAJECTORY_TYPE = "all"
INCLUDE_DISTURBANCES = True
INCLUDE_OBSTACLES = True

ENABLE_OBSTACLES = False
LOOKAHEAD_L = TRACKING_LOOKAHEAD
ENABLE_CBF_AVOIDANCE = False
ENABLE_CBF_DIAGNOSTICS = False
CBF_SAFETY_MARGIN = CBF_MARGIN
CBF_RELATIVE_DEGREE_ALPHA_1 = CBF_ALPHA_1
CBF_RELATIVE_DEGREE_ALPHA_2 = CBF_ALPHA_2
CBF_SELECTED_BARRIER = CBF_BARRIER_TYPE
CBF_SELECTED_ELLIPSE_MAJOR_SCALE = CBF_ELLIPSE_MAJOR_SCALE
CBF_SELECTED_ELLIPSE_MINOR_SCALE = CBF_ELLIPSE_MINOR_SCALE
CBF_SELECTED_ELLIPSE_ANGLE_DEG = CBF_ELLIPSE_ANGLE_DEG
ENABLE_REFERENCE_GOVERNOR = False
REFERENCE_GAMMA_MIN = GOVERNOR_GAMMA_MIN
REFERENCE_GOVERNOR_H_SLOW = GOVERNOR_H_SLOW
REFERENCE_GOVERNOR_H_STOP = GOVERNOR_H_STOP
REFERENCE_GAMMA_FILTER_TAU_MIN = GOVERNOR_FILTER_TAU_MIN
REFERENCE_GAMMA_FILTER_TAU_MAX = GOVERNOR_FILTER_TAU_MAX


DISTURBANCE_OPTIONS = {
    "none": no_disturbance,
    **INERTIAL_DISTURBANCE_CASES,
    "wind": wind_like_disturbance,
    "directional": directional_force_disturbance(
        magnitude=15.0,
        direction_deg=45.0,
    ),
    "constant": constant_disturbance(
        dTu=0.0,
        dTv=0.0,
        dTr=5.0,
    ),
    "sinusoidal": pure_sinusoidal_disturbance,
    "combined": combined_disturbance(
        wind_like_disturbance,
        directional_force_disturbance(
            magnitude=7.0,
            direction_deg=64.0,
        ),
        sinusoidal_disturbance(
            amplitude=(6.0, 3.0, 1.5),
            frequency=(0.04, 0.06, 0.03),
            phase=(0.0, 0.5, 1.0),
        ),
    ),
}
DISTURBANCE_MODE = "none"


def get_trajectory_paths(trajectory_type):
    """
    Devuelve las rutas de salida (CSV y carpetas de figuras) para cada trayectoria:
      - 'figure_eight' -> usa directamente data/figures*, conservando la estructura original
      - 'spiral'       -> usa data/trajectory_spiral/figures*
      - 'clover'       -> usa data/trajectory_clover/figures*
    """
    key = str(trajectory_type).strip().lower()
    if key == TRAJECTORY_FIGURE_EIGHT:
        base_dir = DATA_DIR
        csv_name = "figure_eight_simulation.csv"
        obs_csv_name = "figure_eight_obstacles.csv"
    else:
        base_dir = DATA_DIR / f"trajectory_{key}"
        csv_name = f"{key}_simulation.csv"
        obs_csv_name = f"{key}_obstacles.csv"

    return {
        "base_dir": base_dir,
        "ideal_csv": base_dir / csv_name,
        "ideal_figures": base_dir / "figures",
        "ideal_anim": (base_dir / "figures") / "Trayectory_animation.gif",
        "disturbances_dir": base_dir / "figures_disturbances",
        "obstacles_dir": base_dir / "figures_obstacles",
        "obstacles_csv": (base_dir / "figures_obstacles") / obs_csv_name,
        "obstacles_anim": (base_dir / "figures_obstacles") / "Trayectory_animation.gif",
        "obstacles_disturbances_dir": base_dir / "figures_obstacles_disturbances",
    }


def state_with_reference_rate(state_i, gamma=None):
    """Agrega gamma a un estado aumentado [.., sigma] para evaluar control."""
    if gamma is None or len(state_i) < 7:
        return state_i
    if len(state_i) >= 8:
        out = state_i.copy()
        out[7] = gamma
        return out
    return np.concatenate([state_i, [gamma]])


def controller_output_history(controller, t, state, gamma_history=None):
    """Calcula tau requerido, tau real y fuerzas de propulsores."""
    if hasattr(controller, "output"):
        output = [
            controller.output(
                ti,
                state_with_reference_rate(
                    state_i,
                    None if gamma_history is None else gamma_i,
                ),
            )
            for ti, state_i, gamma_i in zip(
                t,
                state,
                np.ones(len(t)) if gamma_history is None else gamma_history,
            )
        ]
        tau_required, tau_delivered, thruster_forces = (
            np.array(values)
            for values in zip(*output)
        )
    else:
        tau_required = np.array([
            controller.required_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
        tau_delivered = np.array([
            controller.delivered_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
        thruster_forces = np.array([
            controller.thruster_forces(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    return tau_required, tau_delivered, thruster_forces


def reference_history(
    t,
    state=None,
    lookahead_L=LOOKAHEAD_L,
    gamma_history=None,
):
    """Calcula la referencia del centro y del punto adelantado."""
    reference_time = t
    reference_rate = 1.0 if gamma_history is None else gamma_history
    if state is not None and state.shape[1] >= 7:
        reference_time = state[:, 6]

    x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref = (
        figure_eight_reference(reference_time)
    )
    x_dot_ref = reference_rate * x_dot_ref
    y_dot_ref = reference_rate * y_dot_ref
    x_ddot_ref = reference_rate**2 * x_ddot_ref
    y_ddot_ref = reference_rate**2 * y_ddot_ref

    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    u_ref = np.hypot(x_dot_ref, y_dot_ref)
    v_ref = np.zeros_like(u_ref)
    speed_squared = np.maximum(u_ref**2, 1e-12)
    r_ref = (
        x_dot_ref * y_ddot_ref
        - y_dot_ref * x_ddot_ref
    ) / speed_squared

    p_s_ref, p_s_dot_ref, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
    )

    if state is None:
        return (
            x_ref,
            y_ref,
            psi_ref,
            u_ref,
            v_ref,
            r_ref,
            p_s_ref[:, 0],
            p_s_ref[:, 1],
            p_s_dot_ref[:, 0],
            p_s_dot_ref[:, 1],
        )

    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)

    return (
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        v_ref,
        r_ref,
        p_s[:, 0],
        p_s[:, 1],
        p_s_dot[:, 0],
        p_s_dot[:, 1],
        p_s_ref[:, 0],
        p_s_ref[:, 1],
        p_s_dot_ref[:, 0],
        p_s_dot_ref[:, 1],
    )


def save_csv(
    path,
    t,
    state,
    controller=None,
    controller_history=None,
    cbf_history_data=None,
    governor_history_data=None,
    lookahead_L=LOOKAHEAD_L,
):
    """Guarda la simulacion en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data_columns = [t, state[:, 0:6]]
    header_columns = ["t", "x", "y", "psi", "u", "v", "r"]
    gamma_history = None

    if state.shape[1] >= 7:
        data_columns.append(state[:, 6])
        header_columns.append("sigma")

    if governor_history_data is not None:
        gamma_history = governor_history_data["gamma"]
        data_columns.append(gamma_history)
        header_columns.append("gamma")
        data_columns.extend([
            governor_history_data["gamma_target"],
            governor_history_data["tau_gamma"],
        ])
        header_columns.extend([
            "gamma_target",
            "gamma_filter_tau",
        ])

    (
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        v_ref,
        r_ref,
        x_s,
        y_s,
        x_s_dot,
        y_s_dot,
        x_s_ref,
        y_s_ref,
        x_s_dot_ref,
        y_s_dot_ref,
    ) = reference_history(t, state, lookahead_L, gamma_history)
    data_columns.extend([
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        v_ref,
        r_ref,
        x_s,
        y_s,
        x_s_dot,
        y_s_dot,
        x_s_ref,
        y_s_ref,
        x_s_dot_ref,
        y_s_dot_ref,
    ])
    header_columns.extend([
        "x_ref",
        "y_ref",
        "psi_ref",
        "u_ref",
        "v_ref",
        "r_ref",
        "x_s",
        "y_s",
        "x_s_dot",
        "y_s_dot",
        "x_s_ref",
        "y_s_ref",
        "x_s_dot_ref",
        "y_s_dot_ref",
    ])

    if controller is not None and hasattr(controller, "thruster_forces"):
        if controller_history is None:
            controller_history = controller_output_history(controller, t, state)
        tau_required, tau_delivered, thruster_forces = controller_history
        pwm_us = thruster_forces_to_pwm_us(thruster_forces)
        data_columns.extend([
            tau_required[:, 0],
            tau_required[:, 1],
            tau_required[:, 2],
            tau_delivered[:, 0],
            tau_delivered[:, 1],
            tau_delivered[:, 2],
            thruster_forces,
            pwm_us,
        ])
        header_columns.extend([
            "tau_u_cmd",
            "tau_v_cmd",
            "tau_r_cmd",
            "tau_u_real",
            "tau_v_real",
            "tau_r_real",
            "F1_FR",
            "F2_FL",
            "F3_BR",
            "F4_BL",
            "PWM1_FR",
            "PWM2_FL",
            "PWM3_BR",
            "PWM4_BL",
        ])

    if cbf_history_data is not None:
        obstacle_count = cbf_history_data["obstacle_count"]
        data_columns.extend([
            cbf_history_data["h_min"],
            cbf_history_data["active"].astype(float),
            cbf_history_data["z_cbf"][:, 0],
            cbf_history_data["z_cbf"][:, 1],
            cbf_history_data["z_cbf"][:, 2],
            cbf_history_data["delta_z"][:, 0],
            cbf_history_data["delta_z"][:, 1],
            cbf_history_data["delta_z"][:, 2],
        ])
        header_columns.extend([
            "cbf_h_min",
            "cbf_active",
            "cbf_Tu",
            "cbf_Tv",
            "cbf_Tr",
            "cbf_delta_Tu",
            "cbf_delta_Tv",
            "cbf_delta_Tr",
        ])

        for obstacle_index in range(obstacle_count):
            data_columns.append(cbf_history_data["h"][:, obstacle_index])
            header_columns.append(f"cbf_h_obs{obstacle_index + 1}")

    data = np.column_stack(data_columns)
    header = ",".join(header_columns)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def run_single_scenario(
    scenario_label,
    disturbance_fun,
    csv_path,
    figures_dir,
    animation_path,
    trajectory_type=TRAJECTORY_FIGURE_EIGHT,
    enable_obstacles=ENABLE_OBSTACLES,
    enable_cbf_avoidance=ENABLE_CBF_AVOIDANCE,
    enable_cbf_diagnostics=ENABLE_CBF_DIAGNOSTICS,
    enable_reference_governor=ENABLE_REFERENCE_GOVERNOR,
):
    """Ejecuta un escenario de simulacion y guarda CSV, figuras PNG y animacion GIF."""
    set_active_trajectory(trajectory_type)

    print(f"\n============================================================")
    print(f"Trayectoria: {trajectory_type.upper()} | Escenario: {scenario_label}")
    dist_info = getattr(disturbance_fun, "info", None)
    if dist_info is not None:
        print(
            f"  Corriente de agua inercial: Magnitude: {dist_info['magnitude']:.0f} N, "
            f"Angle: {dist_info['direction_deg']:.0f} deg "
            f"(Fx={dist_info['Fx_inertial']:.2f} N, Fy={dist_info['Fy_inertial']:.2f} N)"
        )
    else:
        print("  Corriente de agua: Ninguna (Ideal)")
    print(
        f"  Obstaculos: {'Activos (CBF + Governor)' if enable_obstacles else 'Inactivos'}"
    )
    print(f"============================================================")

    t_final = trajectory_duration(trajectory_type)
    nominal_controller = smc_figure_eight(
        Tu_limits=TU_LIMITS,
        Tv_limits=TV_LIMITS,
        Tr_limits=TR_LIMITS,
        lookahead_L=LOOKAHEAD_L,
        params=PARAMS,
        trajectory_type=trajectory_type,
    )
    state0 = figure_eight_initial_state(trajectory_type=trajectory_type)
    obstacles = configured_obstacles(
        enabled=enable_obstacles,
        trajectory_type=trajectory_type,
    )
    reference_governor_fun = virtual_time_governor(
        obstacles,
        enabled=enable_reference_governor,
        lookahead_L=LOOKAHEAD_L,
        margin=CBF_SAFETY_MARGIN,
        gamma_min=REFERENCE_GAMMA_MIN,
        h_slow=REFERENCE_GOVERNOR_H_SLOW,
        h_stop=REFERENCE_GOVERNOR_H_STOP,
        tau_min=REFERENCE_GAMMA_FILTER_TAU_MIN,
        tau_max=REFERENCE_GAMMA_FILTER_TAU_MAX,
    )
    controller = cbf_filtered_controller(
        nominal_controller,
        obstacles,
        enabled=enable_cbf_avoidance,
        lookahead_L=LOOKAHEAD_L,
        alpha_1=CBF_RELATIVE_DEGREE_ALPHA_1,
        alpha_2=CBF_RELATIVE_DEGREE_ALPHA_2,
        margin=CBF_SAFETY_MARGIN,
        barrier_type=CBF_SELECTED_BARRIER,
        ellipse_major_scale=CBF_SELECTED_ELLIPSE_MAJOR_SCALE,
        ellipse_minor_scale=CBF_SELECTED_ELLIPSE_MINOR_SCALE,
        ellipse_angle_deg=CBF_SELECTED_ELLIPSE_ANGLE_DEG,
    )

    if enable_reference_governor:
        state0 = np.concatenate([state0, [0.0, 1.0]])

    t, state = simulate(
        state0=state0,
        tau_fun=controller,
        disturbance_fun=disturbance_fun,
        t_final=t_final,
        dt=DT,
        params=PARAMS,
        reference_governor_fun=reference_governor_fun,
    )

    governor_history_data = None
    if enable_reference_governor:
        governor_history_data = governor_history(
            state,
            obstacles,
            enabled=enable_reference_governor,
            lookahead_L=LOOKAHEAD_L,
            margin=CBF_SAFETY_MARGIN,
            gamma_min=REFERENCE_GAMMA_MIN,
            h_slow=REFERENCE_GOVERNOR_H_SLOW,
            h_stop=REFERENCE_GOVERNOR_H_STOP,
            tau_min=REFERENCE_GAMMA_FILTER_TAU_MIN,
            tau_max=REFERENCE_GAMMA_FILTER_TAU_MAX,
        )

    gamma_hist = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else None
    )
    controller_history = controller_output_history(
        controller,
        t,
        state,
        gamma_history=gamma_hist,
    )

    cbf_history_data = None
    if enable_cbf_diagnostics:
        cbf_history_data = cbf_history(
            t,
            state,
            controller_history[0],
            obstacles,
            lookahead_L=LOOKAHEAD_L,
            alpha_1=CBF_RELATIVE_DEGREE_ALPHA_1,
            alpha_2=CBF_RELATIVE_DEGREE_ALPHA_2,
            margin=CBF_SAFETY_MARGIN,
            barrier_type=CBF_SELECTED_BARRIER,
            ellipse_major_scale=CBF_SELECTED_ELLIPSE_MAJOR_SCALE,
            ellipse_minor_scale=CBF_SELECTED_ELLIPSE_MINOR_SCALE,
            ellipse_angle_deg=CBF_SELECTED_ELLIPSE_ANGLE_DEG,
        )

    save_csv(
        csv_path,
        t,
        state,
        controller=controller,
        controller_history=controller_history,
        cbf_history_data=cbf_history_data,
        governor_history_data=governor_history_data,
        lookahead_L=LOOKAHEAD_L,
    )
    print(f"Simulacion guardada en: {csv_path}")

    plot_results(
        t,
        state,
        controller=controller,
        controller_history=controller_history,
        Tu_limits=TU_LIMITS,
        Tv_limits=TV_LIMITS,
        Tr_limits=TR_LIMITS,
        disturbance_info=dist_info,
        disturbance_fun=disturbance_fun,
        obstacles=obstacles,
        cbf_history_data=cbf_history_data,
        governor_history_data=governor_history_data,
        lookahead_L=LOOKAHEAD_L,
        save_dir=figures_dir,
        show=SHOW_PLOTS,
    )
    print(f"Graficos guardados en: {figures_dir}")
    for figure_path in sorted(figures_dir.glob("*.png")):
        print(f" - {figure_path}")

    if SAVE_ANIMATION:
        saved_anim = save_simulation_animation(
            t,
            state,
            output_path=animation_path,
            fps=ANIMATION_FPS,
            duration=ANIMATION_DURATION,
            obstacles=obstacles,
            cbf_history_data=cbf_history_data,
            cbf_margin=CBF_SAFETY_MARGIN,
            governor_history_data=governor_history_data,
            lookahead_L=LOOKAHEAD_L,
            disturbance_info=dist_info,
        )
        print(f"Animacion guardada en: {saved_anim}")


def run_trajectory_suite(
    trajectory_type,
    include_disturbances=INCLUDE_DISTURBANCES,
    include_obstacles=INCLUDE_OBSTACLES,
):
    """Ejecuta todos los escenarios configurados para una trayectoria dada."""
    paths = get_trajectory_paths(trajectory_type)

    # 1. Simulacion Ideal (sin corrientes ni obstaculos)
    run_single_scenario(
        scenario_label="Ideal (Sin Corrientes ni Obstaculos)",
        disturbance_fun=no_disturbance,
        csv_path=paths["ideal_csv"],
        figures_dir=paths["ideal_figures"],
        animation_path=paths["ideal_anim"],
        trajectory_type=trajectory_type,
        enable_obstacles=False,
        enable_cbf_avoidance=False,
        enable_cbf_diagnostics=False,
        enable_reference_governor=False,
    )

    # 2. Si INCLUDE_DISTURBANCES es True: corre las 4 corrientes de agua (sin obstaculos)
    if include_disturbances:
        for case_key, dist_fun in INERTIAL_DISTURBANCE_CASES.items():
            case_dir = paths["disturbances_dir"] / case_key
            case_csv = case_dir / f"{case_key}.csv"
            case_anim = case_dir / "Trayectory_animation.gif"
            label = dist_fun.info["name"] if getattr(dist_fun, "info", None) else case_key
            run_single_scenario(
                scenario_label=label,
                disturbance_fun=dist_fun,
                csv_path=case_csv,
                figures_dir=case_dir,
                animation_path=case_anim,
                trajectory_type=trajectory_type,
                enable_obstacles=False,
                enable_cbf_avoidance=False,
                enable_cbf_diagnostics=False,
                enable_reference_governor=False,
            )

    # 3. Si INCLUDE_OBSTACLES es True: corre la simulacion ideal con obstaculos (CBF + Governor)
    if include_obstacles:
        run_single_scenario(
            scenario_label="Ideal con Obstaculos (CBF + Reference Governor, Sin Corrientes)",
            disturbance_fun=no_disturbance,
            csv_path=paths["obstacles_csv"],
            figures_dir=paths["obstacles_dir"],
            animation_path=paths["obstacles_anim"],
            trajectory_type=trajectory_type,
            enable_obstacles=True,
            enable_cbf_avoidance=True,
            enable_cbf_diagnostics=True,
            enable_reference_governor=True,
        )

    # 4. Si AMBAS banderas son True: corre las 4 corrientes de agua CON obstaculos (CBF + Governor)
    if include_disturbances and include_obstacles:
        for case_key, dist_fun in INERTIAL_DISTURBANCE_CASES.items():
            case_dir = paths["obstacles_disturbances_dir"] / case_key
            case_csv = case_dir / f"{case_key}_obstacles.csv"
            case_anim = case_dir / "Trayectory_animation.gif"
            base_label = dist_fun.info["name"] if getattr(dist_fun, "info", None) else case_key
            label = f"{base_label} + Obstaculos (CBF + Governor)"
            run_single_scenario(
                scenario_label=label,
                disturbance_fun=dist_fun,
                csv_path=case_csv,
                figures_dir=case_dir,
                animation_path=case_anim,
                trajectory_type=trajectory_type,
                enable_obstacles=True,
                enable_cbf_avoidance=True,
                enable_cbf_diagnostics=True,
                enable_reference_governor=True,
            )


def main(
    trajectory_type=TRAJECTORY_TYPE,
    include_disturbances=INCLUDE_DISTURBANCES,
    include_obstacles=INCLUDE_OBSTACLES,
):
    print(f"Modelo Fossen 9-param cargado desde: {MODEL_JSON_PATH}")
    print(f"Curva de propulsores y posiciones cargadas desde: {THRUSTERS_JSON_PATH}")

    mode = str(trajectory_type).strip().lower()
    if mode == "all":
        trajectories = (
            TRAJECTORY_FIGURE_EIGHT,
            TRAJECTORY_SPIRAL,
            TRAJECTORY_CLOVER,
        )
    else:
        trajectories = (mode,)

    for traj in trajectories:
        run_trajectory_suite(
            traj,
            include_disturbances=include_disturbances,
            include_obstacles=include_obstacles,
        )


if __name__ == "__main__":
    main()
