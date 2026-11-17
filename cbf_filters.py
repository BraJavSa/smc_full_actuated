import numpy as np

from controllers import TRACKING_LOOKAHEAD, lookahead_point_from_state
from usv_model import PARAMS, allocate_thrusters


CBF_ALPHA_1 = 1.5
CBF_ALPHA_2 = 1.5
CBF_MARGIN = 1.50
CBF_BARRIER_CIRCLE = "circle"
CBF_BARRIER_ELLIPSE = "ellipse"
CBF_BARRIER_COLLISION_CONE = "collision_cone"
CBF_BARRIER_TYPE = CBF_BARRIER_CIRCLE
CBF_ELLIPSE_MAJOR_SCALE = 1.60
CBF_ELLIPSE_MINOR_SCALE = 1.00
CBF_ELLIPSE_ANGLE_DEG = 0.0
CBF_C3BF_EPS = 1e-6
CBF_C3BF_MIN_CONE_CLEARANCE = 0.10


def obstacle_safe_radius(obstacle, margin=CBF_MARGIN):
    """Radio usado por la barrera para un obstaculo circular."""
    return float(obstacle.radius) + margin


def obstacle_motion_state(obstacle, t=None):
    """Obtiene posicion, velocidad y aceleracion plana del obstaculo."""
    if hasattr(obstacle, "position"):
        position = obstacle.position(t)
    else:
        position = (obstacle.x, obstacle.y)

    if hasattr(obstacle, "velocity"):
        velocity = obstacle.velocity(t)
    else:
        velocity = (
            getattr(obstacle, "vx", 0.0),
            getattr(obstacle, "vy", 0.0),
        )

    if hasattr(obstacle, "acceleration"):
        acceleration = obstacle.acceleration(t)
    else:
        acceleration = (
            getattr(obstacle, "ax", 0.0),
            getattr(obstacle, "ay", 0.0),
        )

    return (
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(acceleration, dtype=float),
    )


def ellipse_metric_matrix(angle_deg=CBF_ELLIPSE_ANGLE_DEG):
    """Matriz de rotacion para orientar los ejes de la elipse."""
    angle = np.deg2rad(float(angle_deg))
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s],
        [s,  c],
    ])


def barrier_shape_matrix(
    obstacle,
    barrier_type=CBF_BARRIER_TYPE,
    margin=CBF_MARGIN,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
):
    """
    Devuelve Q y escala para h = rel.T @ Q @ rel - 1.
    """
    safe_radius = obstacle_safe_radius(obstacle, margin)
    barrier_type = str(barrier_type).lower()

    if barrier_type == CBF_BARRIER_CIRCLE:
        return np.eye(2) / safe_radius**2, safe_radius

    if barrier_type == CBF_BARRIER_ELLIPSE:
        major_axis = max(float(ellipse_major_scale) * safe_radius, 1e-9)
        minor_axis = max(float(ellipse_minor_scale) * safe_radius, 1e-9)
        rotation = ellipse_metric_matrix(ellipse_angle_deg)
        metric_body = np.diag([1.0 / major_axis**2, 1.0 / minor_axis**2])
        return rotation @ metric_body @ rotation.T, safe_radius

    raise ValueError(
        "barrier_type debe ser 'circle', 'ellipse' o 'collision_cone', "
        f"pero se recibio {barrier_type!r}."
    )


def lookahead_acceleration_affine(
    state,
    lookahead_L=TRACKING_LOOKAHEAD,
    params=PARAMS,
):
    """
    Calcula p_s_ddot = f_ps + G_ps @ [Tu, Tv, Tr] para el modelo Fossen 3-DOF
    de 9 parametros de PlaDyPos (totalmente actuado en surge, sway y yaw).
    """
    x, y, psi, u, v, r = np.asarray(state, dtype=float)[0:6]
    c = np.cos(psi)
    s = np.sin(psi)

    m11 = params["m11"]
    m22 = params["m22"]
    m33 = params["m33"]
    Xu = params["Xu"]
    Xuu = params["Xuu"]
    Yv = params["Yv"]
    Yvv = params["Yvv"]
    Nr = params["Nr"]
    Nrr = params["Nrr"]

    # Terminos de deriva en aceleracion traslacional y rotacional
    u_dot_drift = (m22 * v * r - (Xu + Xuu * abs(u)) * u) / m11
    v_dot_drift = (-m11 * u * r - (Yv + Yvv * abs(v)) * v) / m22
    r_dot_drift = (-(m22 - m11) * u * v - (Nr + Nrr * abs(r)) * r) / m33

    u_axis_drift = u_dot_drift - v * r
    v_axis_drift = v_dot_drift + u * r

    e_psi = np.array([c, s], dtype=float)
    e_perp = np.array([-s, c], dtype=float)
    rotation = np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)

    center_accel_drift = rotation @ np.array([u_axis_drift, v_axis_drift], dtype=float)
    lookahead_drift = (
        center_accel_drift
        + lookahead_L * r_dot_drift * e_perp
        - lookahead_L * r**2 * e_psi
    )

    # Matriz de entrada G (2x3) que mapea [Tu, Tv, Tr] -> p_s_ddot
    G = np.column_stack([
        (1.0 / m11) * e_psi,
        (1.0 / m22) * e_perp,
        lookahead_L * (1.0 / m33) * e_perp,
    ])

    return lookahead_drift, G


def collision_cone_barrier_constraint(
    state,
    obstacle,
    t=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha=CBF_ALPHA_1,
    margin=CBF_MARGIN,
    params=PARAMS,
):
    """
    C3BF aplicada al punto adelantado con control 3-DOF [Tu, Tv, Tr]:
      h = p_rel.T v_rel + ||v_rel|| sqrt(||p_rel||^2 - r_safe^2)
    """
    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)
    p_s = np.asarray(p_s, dtype=float)
    p_s_dot = np.asarray(p_s_dot, dtype=float)

    obstacle_position, obstacle_velocity, obstacle_acceleration = (
        obstacle_motion_state(obstacle, t)
    )
    p_rel = obstacle_position - p_s
    v_rel = obstacle_velocity - p_s_dot
    safe_radius = obstacle_safe_radius(obstacle, margin)

    distance = float(np.linalg.norm(p_rel))
    speed = float(np.linalg.norm(v_rel))
    cone_clearance_sq = max(
        distance**2 - safe_radius**2,
        CBF_C3BF_MIN_CONE_CLEARANCE**2,
    )
    cone_clearance = np.sqrt(cone_clearance_sq)
    speed_safe = max(speed, CBF_C3BF_EPS)

    h = float(p_rel @ v_rel + speed * cone_clearance)
    grad_p = v_rel + speed * p_rel / cone_clearance
    grad_v = p_rel + cone_clearance * v_rel / speed_safe

    accel_drift, accel_map = lookahead_acceleration_affine(
        state,
        lookahead_L=lookahead_L,
        params=params,
    )
    p_rel_dot = v_rel
    v_rel_drift = obstacle_acceleration - accel_drift
    v_rel_map = -accel_map

    h_dot_drift = float(grad_p @ p_rel_dot + grad_v @ v_rel_drift)
    h_dot_map = grad_v @ v_rel_map

    a = h_dot_map
    b = -(h_dot_drift + alpha * h)

    return {
        "a": np.asarray(a, dtype=float),
        "b": float(b),
        "h": h,
        "h_dot": h_dot_drift,
        "safe_radius": safe_radius,
        "distance": distance,
        "relative_speed": speed,
        "barrier_type": CBF_BARRIER_COLLISION_CONE,
    }


def barrier_constraint(
    state,
    obstacle,
    t=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha_1=CBF_ALPHA_1,
    alpha_2=CBF_ALPHA_2,
    margin=CBF_MARGIN,
    barrier_type=CBF_BARRIER_TYPE,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
    params=PARAMS,
):
    """
    Devuelve la restriccion lineal HOCBF a @ [Tu, Tv, Tr] >= b.
    """
    barrier_type = str(barrier_type).lower()
    if barrier_type == CBF_BARRIER_COLLISION_CONE:
        return collision_cone_barrier_constraint(
            state,
            obstacle,
            t=t,
            lookahead_L=lookahead_L,
            alpha=alpha_1,
            margin=margin,
            params=params,
        )

    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)
    p_s = np.asarray(p_s, dtype=float)
    p_s_dot = np.asarray(p_s_dot, dtype=float)

    center = np.array([obstacle.x, obstacle.y], dtype=float)
    relative_position = p_s - center
    shape_matrix, safe_radius = barrier_shape_matrix(
        obstacle,
        barrier_type=barrier_type,
        margin=margin,
        ellipse_major_scale=ellipse_major_scale,
        ellipse_minor_scale=ellipse_minor_scale,
        ellipse_angle_deg=ellipse_angle_deg,
    )

    shaped_relative_position = shape_matrix @ relative_position
    h = float(relative_position @ shaped_relative_position - 1.0)
    h_dot = float(2.0 * shaped_relative_position @ p_s_dot)

    accel_drift, accel_map = lookahead_acceleration_affine(
        state,
        lookahead_L=lookahead_L,
        params=params,
    )

    h_ddot_drift = 2.0 * (
        p_s_dot @ shape_matrix @ p_s_dot
        + shaped_relative_position @ accel_drift
    )
    h_ddot_map = 2.0 * shaped_relative_position @ accel_map

    a = h_ddot_map
    b = -(
        h_ddot_drift
        + (alpha_1 + alpha_2) * h_dot
        + alpha_1 * alpha_2 * h
    )

    return {
        "a": np.asarray(a, dtype=float),
        "b": float(b),
        "h": h,
        "h_dot": h_dot,
        "safe_radius": safe_radius,
        "distance": float(np.linalg.norm(relative_position)),
        "barrier_type": str(barrier_type).lower(),
    }


def project_control_to_constraints(z_nominal, constraints, tol=1e-9):
    """
    Proyecta el vector de control 3-DOF z = [Tu, Tv, Tr] sobre a_i @ z >= b_i
    minimizando ||z - z_nominal||_2^2.
    """
    z_nominal = np.asarray(z_nominal, dtype=float)
    active_constraints = [
        constraint
        for constraint in constraints
        if np.linalg.norm(constraint["a"]) > tol
    ]

    if not active_constraints:
        return z_nominal.copy(), False

    A = np.vstack([constraint["a"] for constraint in active_constraints])
    b = np.array([constraint["b"] for constraint in active_constraints], dtype=float)

    def is_feasible(z):
        return np.all(A @ z >= b - 1e-7)

    if is_feasible(z_nominal):
        return z_nominal.copy(), False

    candidates = []

    # Proyeccion sobre cada hiperplano individual
    for row, rhs in zip(A, b):
        denom = float(np.dot(row, row))
        if denom <= tol:
            continue
        candidates.append(z_nominal + ((rhs - row @ z_nominal) / denom) * row)

    # Proyeccion de norma minima sobre intersecciones de pares de hiperplanos
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            A_pair = np.vstack([A[i], A[j]])
            b_pair = np.array([b[i], b[j]], dtype=float)
            gram = A_pair @ A_pair.T
            if abs(np.linalg.det(gram)) <= tol:
                continue
            lam = np.linalg.solve(gram, b_pair - A_pair @ z_nominal)
            candidates.append(z_nominal + A_pair.T @ lam)

    feasible_candidates = [candidate for candidate in candidates if is_feasible(candidate)]
    if not feasible_candidates:
        return z_nominal.copy(), False

    z_cbf = min(
        feasible_candidates,
        key=lambda candidate: np.linalg.norm(candidate - z_nominal),
    )
    return z_cbf, True


def cbf_evaluation(
    state,
    z_smc,
    obstacles,
    t=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha_1=CBF_ALPHA_1,
    alpha_2=CBF_ALPHA_2,
    margin=CBF_MARGIN,
    barrier_type=CBF_BARRIER_TYPE,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
):
    """Evalua barreras y calcula el control seguro CBF 3-DOF [Tu, Tv, Tr]."""
    obstacles = () if obstacles is None else tuple(obstacles)
    z_smc = np.asarray(z_smc, dtype=float)
    if z_smc.size == 2:
        z_smc = np.array([z_smc[0], 0.0, z_smc[1]], dtype=float)

    constraints = [
        barrier_constraint(
            state,
            obstacle,
            t=t,
            lookahead_L=lookahead_L,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            margin=margin,
            barrier_type=barrier_type,
            ellipse_major_scale=ellipse_major_scale,
            ellipse_minor_scale=ellipse_minor_scale,
            ellipse_angle_deg=ellipse_angle_deg,
        )
        for obstacle in obstacles
    ]
    z_cbf, active = project_control_to_constraints(z_smc, constraints)

    if constraints:
        h_values = np.array([constraint["h"] for constraint in constraints])
        h_dot_values = np.array([constraint["h_dot"] for constraint in constraints])
        distance_values = np.array([constraint["distance"] for constraint in constraints])
    else:
        h_values = np.empty(0)
        h_dot_values = np.empty(0)
        distance_values = np.empty(0)

    return {
        "z_smc": z_smc,
        "z_cbf": z_cbf,
        "delta_z": z_cbf - z_smc,
        "active": bool(active),
        "h": h_values,
        "h_dot": h_dot_values,
        "distance": distance_values,
        "constraints": constraints,
    }


def cbf_safe_control(
    state,
    z_smc,
    obstacles,
    enabled=True,
    t=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha_1=CBF_ALPHA_1,
    alpha_2=CBF_ALPHA_2,
    margin=CBF_MARGIN,
    barrier_type=CBF_BARRIER_TYPE,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
):
    """Calcula z_safe y Delta z_safe en 3-DOF."""
    evaluation = cbf_evaluation(
        state,
        z_smc,
        obstacles,
        t=t,
        lookahead_L=lookahead_L,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        margin=margin,
        barrier_type=barrier_type,
        ellipse_major_scale=ellipse_major_scale,
        ellipse_minor_scale=ellipse_minor_scale,
        ellipse_angle_deg=ellipse_angle_deg,
    )

    if not enabled:
        z_smc_arr = evaluation["z_smc"]
        evaluation["z_cbf"] = z_smc_arr.copy()
        evaluation["delta_z"] = np.zeros_like(z_smc_arr)
        evaluation["active"] = False

    return evaluation


def cbf_filtered_controller(
    nominal_controller,
    obstacles,
    enabled=True,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha_1=CBF_ALPHA_1,
    alpha_2=CBF_ALPHA_2,
    margin=CBF_MARGIN,
    barrier_type=CBF_BARRIER_TYPE,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
):
    """
    Envuelve un controlador SMC 3-DOF y aplica la CBF sobre [Tu, Tv, Tr]
    antes de asignar a los 4 propulsores en X.
    """
    obstacles = () if obstacles is None else tuple(obstacles)

    def compute_tau(t, state):
        tau_smc, tau_smc_delivered, smc_thruster_forces = (
            nominal_controller.output(t, state)
        )

        if not enabled:
            return tau_smc, tau_smc_delivered, smc_thruster_forces

        z_smc = np.asarray(tau_smc, dtype=float)
        evaluation = cbf_safe_control(
            state,
            z_smc,
            obstacles,
            enabled=True,
            t=t,
            lookahead_L=lookahead_L,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            margin=margin,
            barrier_type=barrier_type,
            ellipse_major_scale=ellipse_major_scale,
            ellipse_minor_scale=ellipse_minor_scale,
            ellipse_angle_deg=ellipse_angle_deg,
        )
        Tu_safe, Tv_safe, Tr_safe = evaluation["z_cbf"]
        tau_applied, thruster_forces = allocate_thrusters(Tu_safe, Tv_safe, Tr_safe)

        return tau_smc, tau_applied, thruster_forces

    def controller(t, state):
        return compute_tau(t, state)[1]

    def required_tau(t, state):
        return compute_tau(t, state)[0]

    def delivered_tau(t, state):
        return compute_tau(t, state)[1]

    def thruster_forces(t, state):
        return compute_tau(t, state)[2]

    def cbf_info(t, state):
        tau_smc = nominal_controller.output(t, state)[0]
        return cbf_safe_control(
            state,
            tau_smc,
            obstacles,
            enabled=enabled,
            t=t,
            lookahead_L=lookahead_L,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            margin=margin,
            barrier_type=barrier_type,
            ellipse_major_scale=ellipse_major_scale,
            ellipse_minor_scale=ellipse_minor_scale,
            ellipse_angle_deg=ellipse_angle_deg,
        )

    controller.output = compute_tau
    controller.required_tau = required_tau
    controller.delivered_tau = delivered_tau
    controller.thruster_forces = thruster_forces
    controller.cbf_info = cbf_info
    controller.nominal_controller = nominal_controller
    controller.cbf_enabled = enabled
    controller.cbf_barrier_type = barrier_type
    controller.lookahead_L = lookahead_L

    return controller


def cbf_history(
    t,
    state,
    tau_smc,
    obstacles,
    lookahead_L=TRACKING_LOOKAHEAD,
    alpha_1=CBF_ALPHA_1,
    alpha_2=CBF_ALPHA_2,
    margin=CBF_MARGIN,
    barrier_type=CBF_BARRIER_TYPE,
    ellipse_major_scale=CBF_ELLIPSE_MAJOR_SCALE,
    ellipse_minor_scale=CBF_ELLIPSE_MINOR_SCALE,
    ellipse_angle_deg=CBF_ELLIPSE_ANGLE_DEG,
):
    """Calcula el historial CBF para el control nominal 3-DOF indicado."""
    evaluations = [
        cbf_evaluation(
            state_i,
            tau_i,
            obstacles,
            t=t_i,
            lookahead_L=lookahead_L,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            margin=margin,
            barrier_type=barrier_type,
            ellipse_major_scale=ellipse_major_scale,
            ellipse_minor_scale=ellipse_minor_scale,
            ellipse_angle_deg=ellipse_angle_deg,
        )
        for t_i, state_i, tau_i in zip(t, state, tau_smc)
    ]

    obstacle_count = len(tuple(()) if obstacles is None else tuple(obstacles))
    if obstacle_count:
        h = np.vstack([evaluation["h"] for evaluation in evaluations])
        h_dot = np.vstack([evaluation["h_dot"] for evaluation in evaluations])
        distance = np.vstack([evaluation["distance"] for evaluation in evaluations])
        h_min = np.min(h, axis=1)
    else:
        h = np.empty((len(t), 0))
        h_dot = np.empty((len(t), 0))
        distance = np.empty((len(t), 0))
        h_min = np.full(len(t), np.nan)

    z_smc = np.vstack([evaluation["z_smc"] for evaluation in evaluations])
    z_cbf = np.vstack([evaluation["z_cbf"] for evaluation in evaluations])
    delta_z = z_cbf - z_smc
    active = np.array([evaluation["active"] for evaluation in evaluations], dtype=bool)

    return {
        "h": h,
        "h_dot": h_dot,
        "h_min": h_min,
        "distance": distance,
        "z_smc": z_smc,
        "z_cbf": z_cbf,
        "delta_z": delta_z,
        "active": active,
        "obstacle_count": obstacle_count,
        "barrier_type": barrier_type,
    }
