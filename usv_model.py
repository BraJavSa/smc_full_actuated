import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


BASE_DIR = Path(__file__).resolve().parent
MODEL_JSON_PATH = BASE_DIR / "model_params.json"
THRUSTERS_JSON_PATH = BASE_DIR / "thrusters_config.json"


def load_model_params(json_path=MODEL_JSON_PATH):
    """Carga el modelo de 9 parametros de Fossen desde el archivo JSON."""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    params = data.get("parameters", data)
    return {
        "m11": float(params["m11"]),
        "m22": float(params["m22"]),
        "m33": float(params["m33"]),
        "Xu": float(params["Xu"]),
        "Xuu": float(params["Xuu"]),
        "Yv": float(params["Yv"]),
        "Yvv": float(params["Yvv"]),
        "Nr": float(params["Nr"]),
        "Nrr": float(params["Nrr"]),
    }


def load_thrusters_config(json_path=THRUSTERS_JSON_PATH):
    """Carga la curva de propulsores (Richards) y sus posiciones en configuracion X desde JSON."""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def build_allocation_matrix_from_config(thrusters_cfg):
    """
    Construye la matriz de asignacion 3x4 (PHI_ALLOC) a partir de las posiciones
    y orientaciones fisicas de cada propulsor definidas en thrusters_config.json.
    Orden de columnas: [M1:FR, M2:FL, M3:BR, M4:BL]
    Filas: [Surge (Tu), Sway (Tv), Yaw (Tr)]
    """
    thrusters = sorted(thrusters_cfg["thrusters"], key=lambda item: item["index"])
    cols = []
    for m in thrusters:
        if "force_direction_body" in m and "moment_arm_m" in m:
            cx = float(m["force_direction_body"]["cos_alpha"])
            sy = float(m["force_direction_body"]["sin_alpha"])
            arm = float(m["moment_arm_m"])
        else:
            x_pos = float(m["position_body_m"]["x"])
            y_pos = float(m["position_body_m"]["y"])
            alpha = math.radians(float(m["azimuth_deg"]))
            cx = math.cos(alpha)
            sy = math.sin(alpha)
            arm = x_pos * sy - y_pos * cx
        cols.append([cx, sy, arm])

    phi_alloc = np.column_stack(cols).astype(float)
    phi_pinv = np.linalg.pinv(phi_alloc)
    return phi_alloc, phi_pinv


# ============================================================
# Carga de parametros desde los archivos JSON
# ============================================================

PARAMS = load_model_params(MODEL_JSON_PATH)
THRUSTERS_CONFIG = load_thrusters_config(THRUSTERS_JSON_PATH)

RICHARDS_PARAMS = THRUSTERS_CONFIG["thruster_curve"]
PWM_CONFIG = THRUSTERS_CONFIG["pwm"]
THRUSTERS_LIST = THRUSTERS_CONFIG["thrusters"]

PHI_ALLOC, PHI_PINV = build_allocation_matrix_from_config(THRUSTERS_CONFIG)

N_THRUSTERS = int(THRUSTERS_CONFIG.get("n_thrusters", 4))
THRUSTER_FWD_MAX = float(RICHARDS_PARAMS["maxForceFwd"])
THRUSTER_REV_MAX = abs(float(RICHARDS_PARAMS["maxForceRev"]))
D_ARM = float(THRUSTERS_CONFIG.get("diagonal_moment_arm_m", 0.65 / math.sqrt(2.0)))

PWM_MIN_US = float(PWM_CONFIG["min_us"])
PWM_NEUTRAL_US = float(PWM_CONFIG["neutral_us"])
PWM_MAX_US = float(PWM_CONFIG["max_us"])
PWM_HALF_RANGE_US = float(PWM_CONFIG["half_range_us"])

# Limites fisicos de esfuerzos generalizados en configuracion X
TU_MAX = float(np.sum(np.abs(PHI_ALLOC[0, :])) * 0.5 * (THRUSTER_FWD_MAX + THRUSTER_REV_MAX))
TU_MIN = -TU_MAX

TV_MAX = float(np.sum(np.abs(PHI_ALLOC[1, :])) * 0.5 * (THRUSTER_FWD_MAX + THRUSTER_REV_MAX))
TV_MIN = -TV_MAX

TR_MAX = float(np.sum(np.abs(PHI_ALLOC[2, :])) * 0.5 * (THRUSTER_FWD_MAX + THRUSTER_REV_MAX))
TR_MIN = -TR_MAX

TU_LIMITS = (TU_MIN, TU_MAX)
TV_LIMITS = (TV_MIN, TV_MAX)
TR_LIMITS = (TR_MIN, TR_MAX)


# ============================================================
# Curva Asimetrica de Richards y Conversion PWM
# ============================================================

def cmd_to_thrust_richards(cmd, richards_params=RICHARDS_PARAMS):
    """
    Calcula el empuje T [N] a partir del comando normalizado cmd en [-1, 1]
    usando la curva logistica generalizada asimetrica (Richards) de PlaDyPos.
    """
    cmd_arr = np.asarray(cmd, dtype=float)
    scalar_input = cmd_arr.ndim == 0
    cmd_s = np.clip(np.atleast_1d(cmd_arr), -1.0, 1.0)
    T = np.zeros_like(cmd_s, dtype=float)

    p = richards_params
    deadzone = float(p.get("deadzone", 0.01))
    pos_mask = cmd_s > deadzone
    neg_mask = cmd_s < -deadzone

    if np.any(pos_mask):
        c_pos = cmd_s[pos_mask]
        denom_pos = (
            p["C_pos"] + np.exp(-p["B_pos"] * (c_pos - p["M_pos"]))
        ) ** (1.0 / p["v_pos"])
        T_pos = p["A_pos"] + (p["K_pos"] - p["A_pos"]) / denom_pos
        T[pos_mask] = np.clip(T_pos, 0.0, p["maxForceFwd"])

    if np.any(neg_mask):
        c_neg = cmd_s[neg_mask]
        denom_neg = (
            p["C_neg"] + np.exp(-p["B_neg"] * (c_neg - p["M_neg"]))
        ) ** (1.0 / p["v_neg"])
        T_neg = p["A_neg"] + (p["K_neg"] - p["A_neg"]) / denom_neg
        T[neg_mask] = np.clip(T_neg, p["maxForceRev"], 0.0)

    if scalar_input:
        return float(T[0])
    return T.reshape(cmd_arr.shape)


def thrust_to_cmd_richards(T_des, richards_params=RICHARDS_PARAMS):
    """
    Inversa analitica de la curva asimetrica de Richards:
    Convierte empuje deseado [N] a comando normalizado en [-1, 1].
    """
    p = richards_params
    T_arr = np.asarray(T_des, dtype=float)
    scalar_input = T_arr.ndim == 0
    T_flat = np.atleast_1d(T_arr).astype(float)
    cmd_out = np.zeros_like(T_flat, dtype=float)

    min_thresh = float(p.get("min_force_threshold_N", 0.15))
    deadzone = float(p.get("deadzone", 0.01))
    max_fwd = float(p["maxForceFwd"])
    max_rev = float(p["maxForceRev"])

    for i, f_val in enumerate(T_flat.flat):
        if abs(f_val) < min_thresh:
            cmd_out.flat[i] = 0.0
        elif f_val >= max_fwd:
            cmd_out.flat[i] = 1.0
        elif f_val <= max_rev:
            cmd_out.flat[i] = -1.0
        elif f_val > 0.0:
            f_c = float(np.clip(f_val, 0.05, max_fwd - 1e-4))
            ratio = (p["K_pos"] - p["A_pos"]) / (f_c - p["A_pos"])
            inner = max(1e-9, (ratio ** p["v_pos"]) - p["C_pos"])
            c = p["M_pos"] - math.log(inner) / p["B_pos"]
            cmd_out.flat[i] = float(np.clip(c, deadzone, 1.0))
        else:
            f_c = float(np.clip(f_val, max_rev + 1e-4, -0.05))
            ratio = (p["K_neg"] - p["A_neg"]) / (f_c - p["A_neg"])
            inner = max(1e-9, (ratio ** p["v_neg"]) - p["C_neg"])
            c = p["M_neg"] - math.log(inner) / p["B_neg"]
            cmd_out.flat[i] = float(np.clip(c, -1.0, -deadzone))

    if scalar_input:
        return float(cmd_out.flat[0])
    return cmd_out.reshape(T_arr.shape)


def thrust_to_normalized_cmd(force):
    """Convierte el empuje de un propulsor [N] a comando normalizado [-1, 1] con la curva de Richards."""
    return thrust_to_cmd_richards(force)


def normalized_cmd_to_pwm_us(cmd):
    """Convierte un comando normalizado [-1, 1] a ancho de pulso PWM [us]."""
    cmd = np.asarray(cmd, dtype=float)
    cmd = np.clip(cmd, -1.0, 1.0)
    return PWM_NEUTRAL_US + PWM_HALF_RANGE_US * cmd


def thrust_to_pwm_us(force):
    """Convierte el empuje de un propulsor [N] a ancho de pulso PWM [us]."""
    return normalized_cmd_to_pwm_us(thrust_to_normalized_cmd(force))


def thruster_forces_to_pwm_us(forces):
    """Convierte fuerzas [F1_FR, F2_FL, F3_BR, F4_BL] [N] a PWM por propulsor [us]."""
    forces = np.asarray(forces, dtype=float)
    return thrust_to_pwm_us(forces)


def allocate_thrusters(Tu_cmd, Tv_cmd=0.0, Tr_cmd=None):
    """
    Asignacion en configuracion X para los 4 propulsores omnidireccionales de PlaDyPos:
      M1: FR (Front Right)
      M2: FL (Front Left)
      M3: BR (Back Right)
      M4: BL (Back Left)

    Soporta llamadas como:
      - allocate_thrusters(Tu, Tv, Tr) -> 3-DOF completo
      - allocate_thrusters([Tu, Tv, Tr]) -> vector 3-DOF
      - allocate_thrusters(Tu, Tr) -> compatibilidad 2 argumentos (Tv = 0)

    Devuelve:
      tau_real    = [Tu_real, Tv_real, Tr_real]
      forces_real = [F1_FR, F2_FL, F3_BR, F4_BL]
    """
    if Tr_cmd is None:
        arr = np.asarray(Tu_cmd, dtype=float)
        if arr.ndim == 1 and arr.size == 3:
            tau_des = arr
        else:
            # Llamada con 2 argumentos: allocate_thrusters(Tu, Tr)
            tau_des = np.array([float(Tu_cmd), 0.0, float(Tv_cmd)], dtype=float)
    else:
        tau_des = np.array([float(Tu_cmd), float(Tv_cmd), float(Tr_cmd)], dtype=float)

    forces_cmd = PHI_PINV @ tau_des
    scale = 1.0
    for f_val in forces_cmd:
        if f_val > THRUSTER_FWD_MAX:
            scale = min(scale, THRUSTER_FWD_MAX / f_val)
        elif f_val < -THRUSTER_REV_MAX:
            scale = min(scale, -THRUSTER_REV_MAX / f_val)
    forces_real = forces_cmd * scale
    tau_real = PHI_ALLOC @ forces_real

    return tau_real, forces_real


# ============================================================
# Cinematica 3-DOF
# eta = [x, y, psi]
# nu  = [u, v, r]
# ============================================================

def j_matrix(psi):
    """
    Matriz de rotacion cinematica 3-DOF del marco cuerpo al marco inercial.
    eta_dot = J(psi) @ nu
    """
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def kinematics(eta, nu):
    """
    eta = [x, y, psi]
    nu  = [u, v, r]
    """
    return j_matrix(eta[2]) @ nu


# ============================================================
# Dinamica Fossen 3-DOF de Caja Blanca (9 Parametros de PlaDyPos)
#
# m11 * u_dot - m22 * v * r + (Xu + Xuu * |u|) * u = Tu
# m22 * v_dot + m11 * u * r + (Yv + Yvv * |v|) * v = Tv
# m33 * r_dot + (m22 - m11) * u * v + (Nr + Nrr * |r|) * r = Tr
# ============================================================

def fossen_coriolis_damping(nu, params=PARAMS):
    """Calcula los vectores de Coriolis C(nu)*nu y amortiguamiento D(nu)*nu."""
    u, v, r = float(nu[0]), float(nu[1]), float(nu[2])
    m11, m22 = params["m11"], params["m22"]
    Xu, Xuu = params["Xu"], params["Xuu"]
    Yv, Yvv = params["Yv"], params["Yvv"]
    Nr, Nrr = params["Nr"], params["Nrr"]

    C_nu = np.array([
        -m22 * v * r,
        m11 * u * r,
        (m22 - m11) * u * v,
    ], dtype=float)

    D_nu = np.array([
        (Xu + Xuu * abs(u)) * u,
        (Yv + Yvv * abs(v)) * v,
        (Nr + Nrr * abs(r)) * r,
    ], dtype=float)

    return C_nu, D_nu


def dynamics_body(nu, tau, params=PARAMS):
    """
    Calcula nu_dot = [u_dot, v_dot, r_dot] para el modelo de Fossen de 9 parametros.

    nu  = [u, v, r]
    tau = [Tu, Tv, Tr]
    """
    u, v, r = float(nu[0]), float(nu[1]), float(nu[2])
    Tu, Tv, Tr = float(tau[0]), float(tau[1]), float(tau[2])

    m11 = params["m11"]
    m22 = params["m22"]
    m33 = params["m33"]
    Xu = params["Xu"]
    Xuu = params["Xuu"]
    Yv = params["Yv"]
    Yvv = params["Yvv"]
    Nr = params["Nr"]
    Nrr = params["Nrr"]

    u_dot = (Tu + m22 * v * r - (Xu + Xuu * abs(u)) * u) / m11
    v_dot = (Tv - m11 * u * r - (Yv + Yvv * abs(v)) * v) / m22
    r_dot = (Tr - (m22 - m11) * u * v - (Nr + Nrr * abs(r)) * r) / m33

    return np.array([u_dot, v_dot, r_dot], dtype=float)


# ============================================================
# Modelo completo en espacio de estados
# state = [x, y, psi, u, v, r] (+ opcionalmente [sigma, gamma])
# ============================================================

def vessel_rhs(
    t,
    state,
    tau_fun,
    disturbance_fun=None,
    params=PARAMS,
    reference_governor_fun=None,
):
    """
    Ecuacion diferencial completa del USV PlaDyPos.

    tau_fun(t, state) devuelve [Tu, Tv, Tr].
    disturbance_fun(t, state) devuelve una perturbacion aditiva [dTu, dTv, dTr].
    """
    vessel_state = state[0:6]
    eta = vessel_state[0:3]
    nu = vessel_state[3:6]

    gamma = None
    gamma_dot = None
    state_for_control = state
    if state.size >= 7:
        if reference_governor_fun is None:
            governor_output = {
                "gamma_target": 1.0,
                "tau_gamma": 1.0,
            }
        else:
            governor_output = reference_governor_fun(t, state)
            if not isinstance(governor_output, dict):
                governor_output = {
                    "gamma_target": float(governor_output),
                    "tau_gamma": 1.0,
                }

        gamma_target = float(governor_output["gamma_target"])
        tau_gamma = max(float(governor_output.get("tau_gamma", 1.0)), 1e-6)
        gamma = state[7] if state.size >= 8 else gamma_target
        gamma_dot = (gamma_target - gamma) / tau_gamma
        state_for_control = np.concatenate([state, [gamma]])

    tau = np.asarray(tau_fun(t, state_for_control), dtype=float)

    if disturbance_fun is not None:
        tau = tau + np.asarray(disturbance_fun(t, state), dtype=float)

    eta_dot = kinematics(eta, nu)
    nu_dot = dynamics_body(nu, tau, params)

    state_dot = np.concatenate([eta_dot, nu_dot])
    if state.size >= 7:
        state_dot = np.concatenate([state_dot, [gamma]])
    if state.size >= 8:
        state_dot = np.concatenate([state_dot, [gamma_dot]])

    return state_dot


def simulate(
    state0,
    tau_fun,
    t_final,
    dt=0.02,
    disturbance_fun=None,
    params=PARAMS,
    reference_governor_fun=None,
    rtol=1e-5,
    atol=1e-7,
):
    """
    Simula la postura y velocidades del USV PlaDyPos.

    Devuelve:
      t     : vector de tiempo
      state : matriz con columnas [x, y, psi, u, v, r] (y sigma, gamma si el governor esta activo)
    """
    t_eval = np.linspace(0.0, t_final, int(round(t_final / dt)) + 1)

    sol = solve_ivp(
        fun=lambda t, x: vessel_rhs(
            t,
            x,
            tau_fun,
            disturbance_fun,
            params,
            reference_governor_fun,
        ),
        t_span=(0.0, t_final),
        y0=state0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T
