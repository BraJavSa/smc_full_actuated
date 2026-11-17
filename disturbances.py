import numpy as np


def no_disturbance(t, state):
    """Corriente nula."""
    return np.zeros(3)


no_disturbance.info = None


def combined_disturbance(*disturbance_funs):
    """Combina varias fuerzas externas aditivas en el cuerpo [Tu, Tv, Tr]."""
    def disturbance_fun(t, state):
        total = np.zeros(3)

        for single_disturbance_fun in disturbance_funs:
            total += np.asarray(single_disturbance_fun(t, state), dtype=float)

        return total

    disturbance_fun.info = None
    return disturbance_fun


def constant_disturbance(dTu=0.0, dTv=0.0, dTr=0.0):
    """Devuelve una fuerza/torque constante en el marco del cuerpo."""
    disturbance = np.array([dTu, dTv, dTr], dtype=float)

    def disturbance_fun(t, state):
        return disturbance

    disturbance_fun.info = None
    return disturbance_fun


def sinusoidal_disturbance(
    amplitude=(0.0, 0.0, 0.0),
    frequency=(0.1, 0.1, 0.1),
    phase=(0.0, 0.0, 0.0),
    bias=(0.0, 0.0, 0.0),
):
    """Devuelve una fuerza externa senoidal aditiva en [Tu, Tv, Tr]."""
    amplitude = np.asarray(amplitude, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    phase = np.asarray(phase, dtype=float)
    bias = np.asarray(bias, dtype=float)

    def disturbance_fun(t, state):
        return bias + amplitude * np.sin(2.0 * np.pi * frequency * t + phase)

    disturbance_fun.info = None
    return disturbance_fun


def pure_sinusoidal_disturbance(t, state):
    """Fuerza senoidal pura sin bias."""
    dTu = 3.0 * np.sin(2.0 * np.pi * 0.035 * t)
    dTv = 4.0 * np.sin(2.0 * np.pi * 0.050 * t + 0.8)
    dTr = 1.5 * np.sin(2.0 * np.pi * 0.025 * t + 1.4)

    return np.array([dTu, dTv, dTr])


pure_sinusoidal_disturbance.info = None


def directional_force_disturbance(magnitude=12.0, direction_deg=30.0, name=""):
    """
    Corriente de agua constante como vector de fuerza fija en el marco INERCIAL:
      F_inertial = [F_x, F_y]^T = magnitude * [cos(direction), sin(direction)]^T

    En cada instante se proyecta al marco del CUERPO (body frame) usando la
    orientacion actual psi = state[2]:
      [dTu, dTv]^T = R(psi)^T @ F_inertial
        dTu =  cos(psi) * F_x + sin(psi) * F_y
        dTv = -sin(psi) * F_x + cos(psi) * F_y
        dTr =  0.0
    """
    direction_rad = np.deg2rad(float(direction_deg))
    force_inertial = np.array([
        float(magnitude) * np.cos(direction_rad),
        float(magnitude) * np.sin(direction_rad),
    ], dtype=float)

    def disturbance_fun(t, state):
        psi = float(state[2])
        c = np.cos(psi)
        s = np.sin(psi)

        dTu = c * force_inertial[0] + s * force_inertial[1]
        dTv = -s * force_inertial[0] + c * force_inertial[1]
        dTr = 0.0

        return np.array([dTu, dTv, dTr], dtype=float)

    disturbance_fun.info = {
        "name": name,
        "magnitude": float(magnitude),
        "direction_deg": float(direction_deg),
        "Fx_inertial": float(force_inertial[0]),
        "Fy_inertial": float(force_inertial[1]),
    }
    return disturbance_fun


def wind_like_disturbance(t, state):
    """Corriente variable en el tiempo."""
    dTu = 2.0 * np.sin(2.0 * np.pi * 0.035 * t)
    dTv = 3.0 + 4.0 * np.sin(2.0 * np.pi * 0.050 * t + 0.8)
    dTr = 1.5 * np.sin(2.0 * np.pi * 0.025 * t + 1.4)

    return np.array([dTu, dTv, dTr])


wind_like_disturbance.info = None


# ============================================================
# 4 Escenarios de Corriente de Agua Constante en el Marco Inercial
# Magnitudes: 12 N, 18 N, 22 N y 30 N en 4 direcciones distintas
# ============================================================
INERTIAL_DISTURBANCE_CASES = {
    "current_1_12N_30deg": directional_force_disturbance(
        magnitude=12.0,
        direction_deg=30.0,
        name="Water Current 1 (Magnitude: 12 N, Angle: 30 deg)",
    ),
    "current_2_18N_135deg": directional_force_disturbance(
        magnitude=18.0,
        direction_deg=135.0,
        name="Water Current 2 (Magnitude: 18 N, Angle: 135 deg)",
    ),
    "current_3_22N_neg60deg": directional_force_disturbance(
        magnitude=22.0,
        direction_deg=-60.0,
        name="Water Current 3 (Magnitude: 22 N, Angle: -60 deg)",
    ),
    "current_4_30N_neg150deg": directional_force_disturbance(
        magnitude=30.0,
        direction_deg=-150.0,
        name="Water Current 4 (Magnitude: 30 N, Angle: -150 deg)",
    ),
}
