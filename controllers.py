import numpy as np

from usv_model import PARAMS, TR_LIMITS, TU_LIMITS, TV_LIMITS, allocate_thrusters


TRAJECTORY_FIGURE_EIGHT = "figure_eight"
TRAJECTORY_SPIRAL       = "spiral"
TRAJECTORY_CLOVER       = "clover"

ACTIVE_TRAJECTORY       = TRAJECTORY_FIGURE_EIGHT

FIGURE_EIGHT_AMPLITUDE  = 15.0
FIGURE_EIGHT_PERIOD     = 380.0      # Permite margen de empuje para corrientes de hasta 30 N
FIGURE_EIGHT_LAPS       = 2.0

# Parametros de la Espiral de Arquimedes
SPIRAL_R_START          = 2.0        # Radio inicial [m]
SPIRAL_R_END            = 15.0       # Radio final [m]
SPIRAL_TURNS            = 2.0        # Numero de vueltas completas
SPIRAL_DURATION         = 760.0      # Duracion total de la espiral [s]

# Parametros del Trebol de 3 Petalos (3-Leaf Clover / Rhodonea)
CLOVER_AMPLITUDE        = 15.0       # Radio maximo de cada petalo [m]
CLOVER_PERIOD           = 840.0      # Periodo para recorrer los 3 petalos [s]
CLOVER_LAPS             = 1.0

TRACKING_LOOKAHEAD      = 0.0        # En 3-DOF full-actuated se controla el centro (L = 0)
REFERENCE_EPS           = 1e-6       # Para evitar division por cero


def set_active_trajectory(trajectory_type):
    """Define la trayectoria activa global ('figure_eight', 'spiral' o 'clover')."""
    global ACTIVE_TRAJECTORY
    key = str(trajectory_type).strip().lower()
    if key not in (TRAJECTORY_FIGURE_EIGHT, TRAJECTORY_SPIRAL, TRAJECTORY_CLOVER):
        raise ValueError(
            f"Trayectoria desconocida: {trajectory_type!r}. "
            "Usa 'figure_eight', 'spiral' o 'clover'."
        )
    ACTIVE_TRAJECTORY = key
    return ACTIVE_TRAJECTORY


def get_active_trajectory():
    """Devuelve el identificador de la trayectoria activa."""
    return ACTIVE_TRAJECTORY


def trajectory_duration(trajectory_type=None):
    """Devuelve el tiempo total de simulacion [s] segun la trayectoria."""
    key = (trajectory_type or ACTIVE_TRAJECTORY).strip().lower()
    if key == TRAJECTORY_SPIRAL:
        return float(SPIRAL_DURATION)
    if key == TRAJECTORY_CLOVER:
        return float(CLOVER_PERIOD * CLOVER_LAPS)
    return float(FIGURE_EIGHT_PERIOD * FIGURE_EIGHT_LAPS)


def trajectory_reference_curve_duration(trajectory_type=None):
    """Duracion para dibujar una curva completa de referencia en las graficas."""
    key = (trajectory_type or ACTIVE_TRAJECTORY).strip().lower()
    if key == TRAJECTORY_SPIRAL:
        return float(SPIRAL_DURATION)
    if key == TRAJECTORY_CLOVER:
        return float(CLOVER_PERIOD)
    return float(FIGURE_EIGHT_PERIOD)


def trajectory_label(trajectory_type=None):
    """Etiqueta legible para la leyenda de la referencia."""
    key = (trajectory_type or ACTIVE_TRAJECTORY).strip().lower()
    if key == TRAJECTORY_SPIRAL:
        return "spiral reference"
    if key == TRAJECTORY_CLOVER:
        return "three-leaf clover reference"
    return "figure-eight reference"


def wrap_angle(angle):
    """Limita un angulo al intervalo [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def reference_time_from_state(t, state):
    """Devuelve sigma si el estado aumentado lo incluye; si no, usa t."""
    state = np.asarray(state, dtype=float)
    if state.shape[-1] >= 7:
        return state[..., 6]

    return t


def reference_rate_from_state(state):
    """Devuelve gamma si fue agregado para la evaluacion del controlador."""
    state = np.asarray(state, dtype=float)
    if state.shape[-1] >= 8:
        return state[..., 7]

    return 1.0


def manual_tau(Tu=0.0, Tv=0.0, Tr=0.0):
    """Devuelve un controlador de fuerzas/torques constantes 3-DOF."""
    def controller(t, state):
        return np.array([Tu, Tv, Tr], dtype=float)

    return controller


def tau_example(t, state):
    """Entrada de prueba por tramos."""
    if t < 10.0:
        return np.array([40.0, 0.0, 0.0], dtype=float)
    if t < 25.0:
        return np.array([30.0, 10.0, 8.0], dtype=float)
    return np.array([0.0, 0.0, 0.0], dtype=float)


def _eval_trajectory_raw(t, trajectory_type=None, amplitude=None, period=None):
    """
    Evalua posicion y hasta 3ra derivada respecto al tiempo de referencia tau = t:
      x, y, dx, dy, ddx, ddy, dddx, dddy
    para la trayectoria seleccionada ('figure_eight', 'spiral' o 'clover').
    """
    key = (trajectory_type or ACTIVE_TRAJECTORY).strip().lower()
    t_arr = np.asarray(t, dtype=float)

    if key == TRAJECTORY_SPIRAL:
        T_tot = float(SPIRAL_DURATION if period is None else period)
        R0 = float(SPIRAL_R_START)
        R1 = float(SPIRAL_R_END if amplitude is None else amplitude)
        a_s = (R1 - R0) / T_tot
        omega_s = 2.0 * np.pi * float(SPIRAL_TURNS) / T_tot

        r = R0 + a_s * t_arr
        th = omega_s * t_arr
        c = np.cos(th)
        s = np.sin(th)

        x_ref = r * c
        y_ref = r * s

        x_dot = a_s * c - r * omega_s * s
        y_dot = a_s * s + r * omega_s * c

        x_ddot = -2.0 * a_s * omega_s * s - r * (omega_s**2) * c
        y_ddot = 2.0 * a_s * omega_s * c - r * (omega_s**2) * s

        x_dddot = -3.0 * a_s * (omega_s**2) * c + r * (omega_s**3) * s
        y_dddot = -3.0 * a_s * (omega_s**2) * s - r * (omega_s**3) * c

        return x_ref, y_ref, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot

    if key == TRAJECTORY_CLOVER:
        # Trebol de 3 petalos (Rosa polar R(th) = A * cos(3*th)):
        #   x = 0.5 * A * (cos(2*w*t) + cos(4*w*t))
        #   y = 0.5 * A * (sin(4*w*t) - sin(2*w*t))
        # Velocidad nunca se anula en ningun punto (v^2 = A^2 w^2 (5 - 4 cos(6 w t)) > 0)
        A = float(CLOVER_AMPLITUDE if amplitude is None else amplitude)
        T_p = float(CLOVER_PERIOD if period is None else period)
        w = 2.0 * np.pi / T_p
        p2 = 2.0 * w * t_arr
        p4 = 4.0 * w * t_arr

        c2, s2 = np.cos(p2), np.sin(p2)
        c4, s4 = np.cos(p4), np.sin(p4)
        half_A = 0.5 * A

        x_ref = half_A * (c2 + c4)
        y_ref = half_A * (s4 - s2)

        x_dot = half_A * w * (-2.0 * s2 - 4.0 * s4)
        y_dot = half_A * w * (4.0 * c4 - 2.0 * c2)

        x_ddot = half_A * (w**2) * (-4.0 * c2 - 16.0 * c4)
        y_ddot = half_A * (w**2) * (-16.0 * s4 + 4.0 * s2)

        x_dddot = half_A * (w**3) * (8.0 * s2 + 64.0 * s4)
        y_dddot = half_A * (w**3) * (-64.0 * c4 + 8.0 * c2)

        return x_ref, y_ref, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot

    # Default: Figure-Eight (Lemniscata)
    A = float(FIGURE_EIGHT_AMPLITUDE if amplitude is None else amplitude)
    T_p = float(FIGURE_EIGHT_PERIOD if period is None else period)
    omega = 2.0 * np.pi / T_p
    phase = omega * t_arr

    x_ref = A * np.sin(phase)
    y_ref = A * np.sin(phase) * np.cos(phase)

    x_dot = A * omega * np.cos(phase)
    y_dot = A * omega * np.cos(2.0 * phase)

    x_ddot = -A * (omega**2) * np.sin(phase)
    y_ddot = -2.0 * A * (omega**2) * np.sin(2.0 * phase)

    x_dddot = -A * (omega**3) * np.cos(phase)
    y_dddot = -4.0 * A * (omega**3) * np.cos(2.0 * phase)

    return x_ref, y_ref, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot


def figure_eight_reference(
    t,
    amplitude=None,
    period=None,
    trajectory_type=None,
):
    """
    Devuelve (x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref)
    para la trayectoria activa ('figure_eight', 'spiral' o 'clover').
    """
    x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref, _, _ = (
        _eval_trajectory_raw(
            t,
            trajectory_type=trajectory_type,
            amplitude=amplitude,
            period=period,
        )
    )
    return x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref


def figure_eight_full_reference(
    t,
    amplitude=None,
    period=None,
    time_scale=1.0,
    eps=REFERENCE_EPS,
    trajectory_type=None,
):
    """
    Calcula la referencia 3-DOF completa en el marco inercial:
      x_ref, y_ref, psi_ref,
      x_dot_ref, y_dot_ref, psi_dot_ref,
      x_ddot_ref, y_ddot_ref, psi_ddot_ref
    """
    (
        x_ref,
        y_ref,
        dx_raw,
        dy_raw,
        ddx_raw,
        ddy_raw,
        dddx_raw,
        dddy_raw,
    ) = _eval_trajectory_raw(
        t,
        trajectory_type=trajectory_type,
        amplitude=amplitude,
        period=period,
    )

    time_scale = np.asarray(time_scale, dtype=float)
    x_dot_ref = time_scale * dx_raw
    y_dot_ref = time_scale * dy_raw

    x_ddot_ref = (time_scale**2) * ddx_raw
    y_ddot_ref = (time_scale**2) * ddy_raw

    x_dddot_ref = (time_scale**3) * dddx_raw
    y_dddot_ref = (time_scale**3) * dddy_raw

    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    v2 = x_dot_ref**2 + y_dot_ref**2 + eps
    num = x_dot_ref * y_ddot_ref - y_dot_ref * x_ddot_ref
    psi_dot_ref = num / v2

    dv2_dt = 2.0 * (x_dot_ref * x_ddot_ref + y_dot_ref * y_ddot_ref)
    dnum_dt = x_dot_ref * y_dddot_ref - y_dot_ref * x_dddot_ref
    psi_ddot_ref = (dnum_dt * v2 - num * dv2_dt) / (v2**2)

    return (
        x_ref,
        y_ref,
        psi_ref,
        x_dot_ref,
        y_dot_ref,
        psi_dot_ref,
        x_ddot_ref,
        y_ddot_ref,
        psi_ddot_ref,
    )


def lookahead_point_from_state(state, lookahead_L=TRACKING_LOOKAHEAD):
    """Calcula el punto de control ps (centro si L=0) y su velocidad inercial ps_dot."""
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    y = state[..., 1]
    psi = state[..., 2]
    u = state[..., 3]
    v = state[..., 4]
    r = state[..., 5]

    c = np.cos(psi)
    s = np.sin(psi)

    x_s = x + lookahead_L * c
    y_s = y + lookahead_L * s
    x_s_dot = c * u - s * v - lookahead_L * r * s
    y_s_dot = s * u + c * v + lookahead_L * r * c

    return (
        np.stack([x_s, y_s], axis=-1),
        np.stack([x_s_dot, y_s_dot], axis=-1),
    )


def figure_eight_lookahead_reference(
    t,
    lookahead_L=TRACKING_LOOKAHEAD,
    time_scale=1.0,
    eps=REFERENCE_EPS,
    trajectory_type=None,
):
    """Referencia del punto de control ps_ref y su velocidad inercial."""
    (
        x_ref,
        y_ref,
        psi_ref,
        x_dot_ref,
        y_dot_ref,
        psi_dot_ref,
        _,
        _,
        _,
    ) = figure_eight_full_reference(
        t,
        time_scale=time_scale,
        eps=eps,
        trajectory_type=trajectory_type,
    )

    c_ref = np.cos(psi_ref)
    s_ref = np.sin(psi_ref)

    x_s_ref = x_ref + lookahead_L * c_ref
    y_s_ref = y_ref + lookahead_L * s_ref
    x_s_dot_ref = x_dot_ref - lookahead_L * psi_dot_ref * s_ref
    y_s_dot_ref = y_dot_ref + lookahead_L * psi_dot_ref * c_ref

    return (
        np.stack([x_s_ref, y_s_ref], axis=-1),
        np.stack([x_s_dot_ref, y_s_dot_ref], axis=-1),
        psi_ref,
        psi_dot_ref,
    )


def figure_eight_initial_state(trajectory_type=None):
    """Estado inicial compatible con el inicio de la trayectoria activa (3-DOF)."""
    (
        x_ref,
        y_ref,
        psi_ref,
        x_dot_ref,
        y_dot_ref,
        psi_dot_ref,
        _,
        _,
        _,
    ) = figure_eight_full_reference(0.0, trajectory_type=trajectory_type)
    u_ref = float(np.hypot(x_dot_ref, y_dot_ref))
    r_ref = float(psi_dot_ref)

    return np.array([
        float(x_ref), float(y_ref), float(psi_ref),
        u_ref, 0.0, r_ref,
    ], dtype=float)


def sat(x):
    """Saturacion suave por componentes en [-1, 1]."""
    return np.clip(x, -1.0, 1.0)


def smc_figure_eight(
    k_pos=0.60,
    k_psi=2.0,
    lambda_u=1.5,
    lambda_v=1.5,
    lambda_r=2.5,
    k_s_u=0.40,
    k_s_v=0.40,
    k_s_r=0.50,
    phi_u=0.05,
    phi_v=0.05,
    phi_r=0.05,
    Tu_limits=TU_LIMITS,
    Tv_limits=TV_LIMITS,
    Tr_limits=TR_LIMITS,
    lookahead_L=TRACKING_LOOKAHEAD,
    params=PARAMS,
    trajectory_type=None,
):
    """
    Controlador por Modo Deslizante (SMC) 3-DOF Totalmente Actuado (Full-Actuated)
    para el USV omnidireccional PlaDyPos (configuracion en X de 4 propulsores).
    Soporta las trayectorias 'figure_eight', 'spiral' y 'clover'.
    """

    def compute_tau(t, state):
        vessel_state = np.asarray(state, dtype=float)[0:6]
        x, y, psi, u, v, r = vessel_state
        reference_time = reference_time_from_state(t, state)
        reference_rate = reference_rate_from_state(state)

        (
            x_ref,
            y_ref,
            psi_ref,
            x_dot_ref,
            y_dot_ref,
            psi_dot_ref,
            x_ddot_ref,
            y_ddot_ref,
            psi_ddot_ref,
        ) = figure_eight_full_reference(
            reference_time,
            time_scale=reference_rate,
            trajectory_type=trajectory_type,
        )

        p_s, p_s_dot = lookahead_point_from_state(vessel_state, lookahead_L)

        c_ref = np.cos(psi_ref)
        s_ref = np.sin(psi_ref)
        e_psi_ref = np.array([c_ref, s_ref], dtype=float)
        e_perp_ref = np.array([-s_ref, c_ref], dtype=float)

        p_ref = np.array([x_ref, y_ref], dtype=float)
        p_dot_ref = np.array([x_dot_ref, y_dot_ref], dtype=float)
        p_ddot_ref = np.array([x_ddot_ref, y_ddot_ref], dtype=float)

        p_s_ref = p_ref + lookahead_L * e_psi_ref
        p_s_dot_ref = p_dot_ref + lookahead_L * psi_dot_ref * e_perp_ref
        p_s_ddot_ref = (
            p_ddot_ref
            + lookahead_L * psi_ddot_ref * e_perp_ref
            - lookahead_L * (psi_dot_ref**2) * e_psi_ref
        )

        # ------------------------------------------------------------
        # 1. Errores y velocidades deseadas en el marco inercial
        # ------------------------------------------------------------
        e_p = p_s_ref - p_s
        e_v = p_s_dot_ref - p_s_dot

        v_cmd_inertial = p_s_dot_ref + k_pos * e_p
        v_cmd_inertial_dot = p_s_ddot_ref + k_pos * e_v

        # Seguimiento directo de la orientacion tangente psi_ref (3-DOF independiente)
        e_psi = wrap_angle(psi_ref - psi)
        e_r = psi_dot_ref - r
        r_cmd = psi_dot_ref + k_psi * e_psi
        r_dot_cmd = psi_ddot_ref + k_psi * e_r

        # ------------------------------------------------------------
        # 2. Proyeccion cinematica al marco cuerpo: J(psi)^T
        # ------------------------------------------------------------
        c = np.cos(psi)
        s = np.sin(psi)

        u_b = c * v_cmd_inertial[0] + s * v_cmd_inertial[1]
        v_b = -s * v_cmd_inertial[0] + c * v_cmd_inertial[1]

        u_b_dot = (
            c * v_cmd_inertial_dot[0]
            + s * v_cmd_inertial_dot[1]
            + r * v_b
        )
        v_b_dot = (
            -s * v_cmd_inertial_dot[0]
            + c * v_cmd_inertial_dot[1]
            - r * u_b
        )

        u_cmd = u_b
        v_cmd = v_b - lookahead_L * r_cmd

        u_dot_cmd = u_b_dot
        v_dot_cmd = v_b_dot - lookahead_L * r_dot_cmd

        # ------------------------------------------------------------
        # 3. Superficies de deslizamiento 3-DOF (Surge, Sway, Yaw)
        # ------------------------------------------------------------
        s_u = u - u_cmd
        s_v = v - v_cmd
        s_r = r - r_cmd

        # ------------------------------------------------------------
        # 4. Ley SMC en aceleraciones deseadas del cuerpo
        # ------------------------------------------------------------
        u_dot_des = (
            u_dot_cmd
            - lambda_u * s_u
            - k_s_u * sat(s_u / phi_u)
        )
        v_dot_des = (
            v_dot_cmd
            - lambda_v * s_v
            - k_s_v * sat(s_v / phi_v)
        )
        r_dot_des = (
            r_dot_cmd
            - lambda_r * s_r
            - k_s_r * sat(s_r / phi_r)
        )

        # ------------------------------------------------------------
        # 5. Inversion del modelo Fossen 3-DOF de 9 parametros (PlaDyPos)
        # ------------------------------------------------------------
        m11 = params["m11"]
        m22 = params["m22"]
        m33 = params["m33"]
        Xu = params["Xu"]
        Xuu = params["Xuu"]
        Yv = params["Yv"]
        Yvv = params["Yvv"]
        Nr = params["Nr"]
        Nrr = params["Nrr"]

        Tu = (Xu + Xuu * abs(u)) * u - m22 * v * r + m11 * u_dot_des
        Tv = (Yv + Yvv * abs(v)) * v + m11 * u * r + m22 * v_dot_des
        Tr = (Nr + Nrr * abs(r)) * r + (m22 - m11) * u * v + m33 * r_dot_des

        tau_required = np.array([Tu, Tv, Tr], dtype=float)
        tau_delivered, thruster_forces = allocate_thrusters(Tu, Tv, Tr)

        return tau_required, tau_delivered, thruster_forces

    def controller(t, state):
        return compute_tau(t, state)[1]

    def required_tau(t, state):
        return compute_tau(t, state)[0]

    def delivered_tau(t, state):
        return compute_tau(t, state)[1]

    def thruster_forces(t, state):
        return compute_tau(t, state)[2]

    controller.output = compute_tau
    controller.required_tau = required_tau
    controller.delivered_tau = delivered_tau
    controller.thruster_forces = thruster_forces
    controller.lookahead_L = lookahead_L

    return controller
