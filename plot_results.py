import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon

from cbf_filters import CBF_MARGIN
from controllers import (
    FIGURE_EIGHT_PERIOD,
    TRACKING_LOOKAHEAD,
    figure_eight_lookahead_reference,
    figure_eight_reference,
    get_active_trajectory,
    lookahead_point_from_state,
    trajectory_label,
    trajectory_reference_curve_duration,
)
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX, THRUSTERS_CONFIG

VESSEL_SIDE_M = float(THRUSTERS_CONFIG.get("thruster_separation_L_m", 0.65))


def _transform_points(points_body, x, y, psi):
    """Rota y traslada un conjunto de puntos 2D del marco cuerpo al inercial."""
    c = np.cos(psi)
    s = np.sin(psi)
    rotation = np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)
    points_world = np.asarray(points_body, dtype=float) @ rotation.T
    points_world[:, 0] += x
    points_world[:, 1] += y
    return points_world


def pladypos_symbol_geometry(x, y, psi, size=VESSEL_SIDE_M):
    """
    Construye la geometria del simbolo omnidireccional en configuracion X
    con las proporciones fisicas reales de PlaDyPos (lado L = 0.65 m entre
    centros de propulsores, es decir, motores en x = +-0.325 m, y = +-0.325 m).

    El eje +x del cuerpo apunta hacia el FRENTE (proa).
    """
    L = float(size)
    a = 0.50 * L          # 0.325 m cuando L = 0.65 m (65 cm de lado entre motores)
    c = 0.22 * L          # Semiancho del cuerpo central cuadrado
    w_arm = 0.062 * L     # Semiancho del brazo al llegar al motor
    d_arm = w_arm * np.sqrt(2.0)

    # Poligono del cuerpo central + 4 brazos diagonales en X (sentido antihorario)
    frame_body = np.array([
        # Lado frontal (+x)
        [c, -c * 0.65],
        # Brazo Front-Right (+x, -y)
        [a - d_arm, -a],
        [a, -a + d_arm],
        # Lado derecho (-y)
        [c * 0.65, -c],
        [-c * 0.65, -c],
        # Brazo Back-Right (-x, -y)
        [-a, -a + d_arm],
        [-a + d_arm, -a],
        # Lado trasero (-x)
        [-c, -c * 0.65],
        [-c, c * 0.65],
        # Brazo Back-Left (-x, +y)
        [-a + d_arm, a],
        [-a, a - d_arm],
        # Lado izquierdo (+y)
        [-c * 0.65, c],
        [c * 0.65, c],
        # Brazo Front-Left (+x, +y)
        [a, a - d_arm],
        [a - d_arm, a],
        # Cierre lado frontal (+x)
        [c, c * 0.65],
        [c, -c * 0.65],
    ], dtype=float)

    # Centros de los 4 motores en (+-0.325, +-0.325) m y angulos de esquina
    corners = [
        (a, -a, -45.0),    # M1: FR
        (a,  a,  45.0),    # M2: FL
        (-a, -a, -135.0),  # M3: BR
        (-a,  a,  135.0),  # M4: BL
    ]

    r_hub = 0.090 * L
    r_ring_in = 0.195 * L
    r_ring_out = 0.260 * L
    theta_hub = np.linspace(0.0, 2.0 * np.pi, 20)

    hubs_world = []
    rings_world = []
    line_segments = [_transform_points(frame_body, x, y, psi)]

    for cx, cy, ang_deg in corners:
        # Hub circular del motor
        hub_pts = np.column_stack([
            cx + r_hub * np.cos(theta_hub),
            cy + r_hub * np.sin(theta_hub),
        ])
        hub_w = _transform_points(hub_pts, x, y, psi)
        hubs_world.append(hub_w)

        # Arco protector C-ring (250 grados alrededor del exterior de la esquina)
        ang_center = np.deg2rad(ang_deg)
        arc_angles = np.linspace(ang_center - np.deg2rad(125.0), ang_center + np.deg2rad(125.0), 28)
        outer_arc = np.column_stack([
            cx + r_ring_out * np.cos(arc_angles),
            cy + r_ring_out * np.sin(arc_angles),
        ])
        inner_arc = np.column_stack([
            cx + r_ring_in * np.cos(arc_angles[::-1]),
            cy + r_ring_in * np.sin(arc_angles[::-1]),
        ])
        ring_poly = np.vstack([outer_arc, inner_arc, outer_arc[0]])
        ring_w = _transform_points(ring_poly, x, y, psi)
        rings_world.append(ring_w)

        line_segments.append(np.array([[np.nan, np.nan]]))
        line_segments.append(ring_w)

    # Pieza frontal ("la cosita del frente" en +x, cara exterior lisa/redondeada en bx1
    # y las dos muescas mirando hacia el cuerpo central en bx0, igual que en la imagen)
    bx0 = c + 0.040 * L
    bx1 = c + 0.200 * L
    by = 0.185 * L
    front_body = np.array([
        [bx1, -by * 0.85],
        [bx1,  by * 0.85],
        [bx0 + 0.035 * L,  by],
        [bx0,  by],
        [bx0,  by * 0.48],
        [bx0 + 0.090 * L,  by * 0.48],
        [bx0 + 0.090 * L,  by * 0.20],
        [bx0,  by * 0.20],
        [bx0, -by * 0.20],
        [bx0 + 0.090 * L, -by * 0.20],
        [bx0 + 0.090 * L, -by * 0.48],
        [bx0, -by * 0.48],
        [bx0, -by],
        [bx0 + 0.035 * L, -by],
        [bx1, -by * 0.85],
    ], dtype=float)
    front_world = _transform_points(front_body, x, y, psi)

    return {
        "frame_poly": line_segments[0],
        "hubs": hubs_world,
        "rings": rings_world,
        "front_bumper": front_world,
        "outline_line": np.vstack(line_segments),
    }


def boat_outline(x, y, psi, length=1.2, width=0.55):
    """
    Devuelve los puntos del contorno en X del bote omnidireccional PlaDyPos.
    """
    geom = pladypos_symbol_geometry(x, y, psi, size=length)
    return geom["outline_line"]


def draw_boat(
    ax,
    x,
    y,
    psi,
    length=1.2,
    width=0.55,
    color="tab:blue",
    front_color="#FF2A00",
    alpha=0.85,
    linewidth=1.2,
    label=None,
    zorder=4,
    **kwargs,
):
    """
    Dibuja el simbolo omnidireccional en X de PlaDyPos con sus 4 propulsores
    y la pieza frontal en un color diferente (`front_color`) para indicar el frente.
    """
    geom = pladypos_symbol_geometry(x, y, psi, size=length)

    frame_patch = Polygon(
        geom["frame_poly"],
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth * 0.6,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(frame_patch)

    for hub_w in geom["hubs"]:
        ax.add_patch(
            Polygon(
                hub_w,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=linewidth * 0.5,
                zorder=zorder,
            )
        )

    for ring_w in geom["rings"]:
        ax.add_patch(
            Polygon(
                ring_w,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=linewidth * 0.5,
                zorder=zorder,
            )
        )

    # Pieza frontal en color distinto para identificar claramente la proa (+x)
    front_patch = Polygon(
        geom["front_bumper"],
        closed=True,
        facecolor=front_color,
        edgecolor="black" if alpha > 0.6 else front_color,
        alpha=min(1.0, alpha + 0.15),
        linewidth=max(0.5, linewidth * 0.45),
        zorder=zorder + 0.2,
    )
    ax.add_patch(front_patch)


def draw_obstacles(ax, obstacles):
    """Dibuja boyas circulares y el contorno ocupado por cada obstaculo."""
    for index, obstacle in enumerate(obstacles):
        boundary_label = "obstacle boundary" if index == 0 else "_nolegend_"
        center_label = "buoy center" if index == 0 else "_nolegend_"
        boundary = Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            edgecolor="magenta",
            facecolor="magenta",
            alpha=0.25,
            linewidth=2.0,
            label=boundary_label,
            zorder=2,
        )
        ax.add_patch(boundary)
        ax.scatter(
            obstacle.x,
            obstacle.y,
            color="magenta",
            edgecolors="black",
            linewidths=0.5,
            s=28,
            label=center_label,
            zorder=4,
        )


def draw_disturbance_arrow(ax, magnitude, direction_deg):
    """Dibuja la flecha y el recuadro de corriente de agua abajo a la izquierda."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    arrow_length = 0.11 * min(x_span, y_span)

    direction_rad = np.deg2rad(direction_deg)
    dx = arrow_length * np.cos(direction_rad)
    dy = arrow_length * np.sin(direction_rad)

    # Ubicacion abajo a la izquierda (en la esquina inferior izquierda libre)
    box_x = x_limits[0] + 0.03 * x_span
    box_y = y_limits[0] + 0.03 * y_span

    # Punto central de la flecha a la derecha del recuadro de texto
    arrow_cx = x_limits[0] + 0.34 * x_span
    arrow_cy = y_limits[0] + 0.085 * y_span
    x0 = arrow_cx - 0.5 * dx
    y0 = arrow_cy - 0.5 * dy

    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "->",
            "color": "#0284C7",
            "linewidth": 2.2,
        },
        zorder=7,
    )
    ax.text(
        box_x,
        box_y,
        f"Water Current\nMagnitude: {magnitude:.0f} N\nAngle: {direction_deg:.0f}°",
        color="#0369A1",
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#0284C7",
            "alpha": 0.92,
        },
        zorder=7,
    )


def plot_disturbance(t, state, disturbance_fun):
    """Grafica la fuerza de corriente de agua proyectada en el cuerpo [Tu, Tv, Tr]."""
    disturbance = np.array([
        disturbance_fun(ti, state_i)
        for ti, state_i in zip(t, state)
    ])

    fig, ax = plt.subplots()
    ax.plot(t, disturbance[:, 0], label="Current Surge Force (body X) [N]")
    ax.plot(t, disturbance[:, 1], label="Current Sway Force (body Y) [N]")
    ax.plot(t, disturbance[:, 2], label="Current Yaw Moment (body Z) [N m]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water current force in body frame")
    info = getattr(disturbance_fun, "info", None)
    if info is not None:
        ax.set_title(
            f"Water Current Projected to Body Frame "
            f"(Magnitude: {info['magnitude']:.0f} N, Angle: {info['direction_deg']:.0f}°)"
        )
    else:
        ax.set_title("Water Current Forces in Body Frame")
    ax.legend()
    ax.grid(True)

    return fig


def plot_tracking_error(
    t,
    state,
    lookahead_L=TRACKING_LOOKAHEAD,
    governor_history_data=None,
):
    """Grafica errores de posicion y velocidad del punto adelantado."""
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )
    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)
    p_s_ref, p_s_dot_ref, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
    )

    error = p_s_ref - p_s
    error_x = error[:, 0]
    error_y = error[:, 1]
    position_error_norm = np.hypot(error_x, error_y)

    velocity_error = p_s_dot_ref - p_s_dot
    velocity_error_x = velocity_error[:, 0]
    velocity_error_y = velocity_error[:, 1]
    velocity_error_norm = np.hypot(velocity_error_x, velocity_error_y)

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, position_error_norm, label="||e_p||")
    axes[0].plot(t, error_x, "--", alpha=0.75, label="e_x")
    axes[0].plot(t, error_y, "--", alpha=0.75, label="e_y")
    axes[0].set_ylabel("position error [m]")
    axes[0].set_title("Lookahead Point Position Tracking Error")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, velocity_error_norm, label="||e_v||")
    axes[1].plot(t, velocity_error_x, "--", alpha=0.75, label="e_vx")
    axes[1].plot(t, velocity_error_y, "--", alpha=0.75, label="e_vy")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("velocity error [m/s]")
    axes[1].set_title("Lookahead Point Velocity Tracking Error")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    return fig


def plot_torque_saturation(
    t,
    state,
    controller,
    Tu_limits=None,
    Tv_limits=None,
    Tr_limits=None,
    tau_required=None,
    tau_delivered=None,
):
    """Grafica las entradas 3-DOF pedidas y entregadas (Tu, Tv, Tr)."""
    if tau_required is None and hasattr(controller, "required_tau"):
        tau_required = np.array([
            controller.required_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
    elif tau_required is None:
        tau_required = np.array([
            controller(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    if tau_delivered is None and hasattr(controller, "delivered_tau"):
        tau_delivered = np.array([
            controller.delivered_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
    elif tau_delivered is None:
        tau_delivered = np.array([
            controller(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.2))

    axes[0].plot(t, tau_required[:, 0], label="required surge Tu", alpha=0.75)
    axes[0].plot(t, tau_delivered[:, 0], label="delivered surge Tu", linewidth=1.8)
    if Tu_limits is not None:
        axes[0].axhline(Tu_limits[0], color="black", linestyle=":", linewidth=1.1, label="limits")
        axes[0].axhline(Tu_limits[1], color="black", linestyle=":", linewidth=1.1)
    axes[0].set_ylabel("Tu [N]")
    axes[0].set_title("Required vs Delivered Surge Force (Tu)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, tau_required[:, 1], label="required sway Tv", alpha=0.75)
    axes[1].plot(t, tau_delivered[:, 1], label="delivered sway Tv", linewidth=1.8)
    if Tv_limits is not None:
        axes[1].axhline(Tv_limits[0], color="black", linestyle=":", linewidth=1.1, label="limits")
        axes[1].axhline(Tv_limits[1], color="black", linestyle=":", linewidth=1.1)
    axes[1].set_ylabel("Tv [N]")
    axes[1].set_title("Required vs Delivered Sway Force (Tv)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, tau_required[:, 2], label="required yaw Tr", alpha=0.75)
    axes[2].plot(t, tau_delivered[:, 2], label="delivered yaw Tr", linewidth=1.8)
    if Tr_limits is not None:
        axes[2].axhline(Tr_limits[0], color="black", linestyle=":", linewidth=1.1, label="limits")
        axes[2].axhline(Tr_limits[1], color="black", linestyle=":", linewidth=1.1)
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("Tr [N m]")
    axes[2].set_title("Required vs Delivered Yaw Torque (Tr)")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()

    return fig


def plot_thruster_forces(t, state, controller, forces=None):
    """Grafica la fuerza entregada por cada propulsor en configuracion X."""
    if forces is None and not hasattr(controller, "thruster_forces"):
        raise ValueError("The controller does not expose thruster_forces(t, state).")

    if forces is None:
        forces = np.array([
            controller.thruster_forces(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    fig, ax = plt.subplots()
    ax.plot(t, forces[:, 0], label="M1 FR (Front Right)")
    ax.plot(t, forces[:, 1], label="M2 FL (Front Left)")
    ax.plot(t, forces[:, 2], label="M3 BR (Back Right)")
    ax.plot(t, forces[:, 3], label="M4 BL (Back Left)")
    ax.axhline(
        THRUSTER_FWD_MAX,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Richards limits (+36.38 / -28.44 N)",
    )
    ax.axhline(
        -THRUSTER_REV_MAX,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("thruster force [N]")
    ax.set_title("Individual X-Thruster Forces (PlaDyPos)")
    ax.legend()
    ax.grid(True)

    return fig


def plot_cbf_barrier(t, cbf_history_data):
    """Grafica la evolucion de la funcion de barrera por obstaculo."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    h = cbf_history_data["h"]
    h_min = cbf_history_data["h_min"]
    active = cbf_history_data["active"].astype(float)

    if h.shape[1] == 0:
        axes[0].plot(t, np.zeros_like(t), label="no obstacles")
    else:
        for obstacle_index in range(h.shape[1]):
            axes[0].plot(
                t,
                h[:, obstacle_index],
                label=f"h obstacle {obstacle_index + 1}",
            )
        axes[0].plot(t, h_min, "k--", linewidth=1.5, label="min h")

    axes[0].axhline(0.0, color="red", linestyle=":", linewidth=1.2, label="h = 0")
    axes[0].set_ylabel("barrier h")
    axes[0].set_title("CBF Barrier Function at Lookahead Point")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, active, color="tab:red", label="CBF active")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("active")
    axes[1].set_yticks([0.0, 1.0])
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    return fig


def plot_reference_governor(t, governor_history_data):
    """Grafica el tiempo virtual sigma y la velocidad gamma."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    sigma = governor_history_data["sigma"]
    gamma = governor_history_data["gamma"]
    gamma_target = governor_history_data.get("gamma_target")
    tau_gamma = governor_history_data.get("tau_gamma")

    if sigma is not None:
        axes[0].plot(t, sigma, label="sigma")
        axes[0].plot(t, t, "k--", alpha=0.35, label="real time")
    else:
        axes[0].plot(t, t, label="sigma = t")

    axes[0].set_ylabel("reference time [s]")
    axes[0].set_title("Virtual-Time Reference Governor")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, gamma, color="tab:orange", label="gamma")
    if gamma_target is not None:
        axes[1].plot(
            t,
            gamma_target,
            "k--",
            alpha=0.45,
            label="gamma target",
        )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("sigma_dot")
    axes[1].legend()
    axes[1].grid(True)

    if tau_gamma is not None:
        ax_tau = axes[1].twinx()
        ax_tau.plot(
            t,
            tau_gamma,
            color="tab:blue",
            alpha=0.45,
            label="filter tau",
        )
        ax_tau.set_ylabel("filter tau [s]")
        ax_tau.legend(loc="lower right")

    fig.tight_layout()

    return fig


def save_simulation_animation(
    t,
    state,
    output_path,
    fps=25,
    duration=20.0,
    show_reference=True,
    obstacles=None,
    cbf_history_data=None,
    cbf_margin=CBF_MARGIN,
    governor_history_data=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    disturbance_info=None,
):
    """Guarda una animacion del movimiento del USV PlaDyPos y su referencia."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = state[:, 0]
    y = state[:, 1]
    psi = state[:, 2]
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )
    x_ref, y_ref, x_dot_ref, y_dot_ref, _, _ = figure_eight_reference(
        reference_time
    )
    x_dot_ref = reference_rate * x_dot_ref
    y_dot_ref = reference_rate * y_dot_ref
    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    p_s, _ = lookahead_point_from_state(state, lookahead_L)
    p_s_ref, _, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
    )
    ref_curve_duration = trajectory_reference_curve_duration()
    t_one_lap = np.linspace(0.0, ref_curve_duration, 700)
    x_path_ref, y_path_ref, _, _, _, _ = figure_eight_reference(t_one_lap)
    p_s_path_ref, _, _, _ = figure_eight_lookahead_reference(t_one_lap, lookahead_L)

    frame_count = max(2, int(fps * duration))
    frame_indices = np.linspace(0, len(t) - 1, frame_count, dtype=int)

    obstacles = () if obstacles is None else obstacles
    x_all = np.concatenate([x, x_path_ref, p_s[:, 0], p_s_path_ref[:, 0]])
    y_all = np.concatenate([y, y_path_ref, p_s[:, 1], p_s_path_ref[:, 1]])
    if obstacles:
        x_obstacle_bounds = np.array([
            coordinate
            for obstacle in obstacles
            for coordinate in (
                obstacle.x - obstacle.radius - cbf_margin,
                obstacle.x + obstacle.radius + cbf_margin,
            )
        ])
        y_obstacle_bounds = np.array([
            coordinate
            for obstacle in obstacles
            for coordinate in (
                obstacle.y - obstacle.radius - cbf_margin,
                obstacle.y + obstacle.radius + cbf_margin,
            )
        ])
        x_all = np.concatenate([x_all, x_obstacle_bounds])
        y_all = np.concatenate([y_all, y_obstacle_bounds])
    span = max(np.ptp(x_all), np.ptp(y_all), 1.0)
    margin = 0.12 * span
    boat_size = VESSEL_SIDE_M

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.set_xlim(np.min(x_all) - margin, np.max(x_all) + margin)
    ax.set_ylim(np.min(y_all) - margin, np.max(y_all) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("PlaDyPos USV Full-Actuated SMC Simulation")
    ax.grid(True)

    if show_reference:
        ax.plot(
            x_path_ref,
            y_path_ref,
            linestyle="--",
            dashes=(4, 4),
            color="0.25",
            alpha=0.85,
            linewidth=1.3,
            label=trajectory_label(),
            zorder=3,
        )

    cbf_safety_patches = []
    if obstacles:
        for index, obstacle in enumerate(obstacles):
            label = "CBF safety zone" if index == 0 else "_nolegend_"
            safety_zone = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius + cbf_margin,
                edgecolor=(1.0, 0.62, 0.0, 0.35),
                facecolor=(1.0, 0.85, 0.0, 0.10),
                linewidth=1.4,
                label=label,
                zorder=1,
            )
            ax.add_patch(safety_zone)
            cbf_safety_patches.append(safety_zone)

        draw_obstacles(ax, obstacles)

    actual_path, = ax.plot(
        [], [], color="tab:blue", linewidth=1.8, label="USV path", zorder=2
    )
    reference_point, = ax.plot(
        [],
        [],
        "o",
        color="#00C853",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=3.6,
        label="reference point",
        zorder=6,
    )
    actual_boat, = ax.plot([], [], color="#111827", linewidth=1.8, zorder=4)
    actual_front, = ax.plot([], [], color="#FF2A00", linewidth=2.2, zorder=5)
    cbf_text = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="0.15",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    ax.legend(loc="upper right", fontsize=8.5)

    if disturbance_info is not None:
        draw_disturbance_arrow(
            ax,
            magnitude=disturbance_info["magnitude"],
            direction_deg=disturbance_info["direction_deg"],
        )

    def update(frame_index):
        index = frame_indices[frame_index]
        cbf_active = False
        critical_obstacle = None

        if (
            cbf_history_data is not None
            and cbf_history_data["h"].shape[1] > 0
        ):
            cbf_active = bool(cbf_history_data["active"][index])
            critical_obstacle = int(np.argmin(cbf_history_data["h"][index]))

        actual_path.set_data(x[:index + 1], y[:index + 1])
        reference_point.set_data([x_ref[index]], [y_ref[index]])

        geom_actual = pladypos_symbol_geometry(
            x[index],
            y[index],
            psi[index],
            size=boat_size,
        )
        actual_boat.set_data(
            geom_actual["outline_line"][:, 0],
            geom_actual["outline_line"][:, 1],
        )
        actual_front.set_data(
            geom_actual["front_bumper"][:, 0],
            geom_actual["front_bumper"][:, 1],
        )

        for obstacle_index, safety_zone in enumerate(cbf_safety_patches):
            is_critical = cbf_active and obstacle_index == critical_obstacle
            if is_critical:
                safety_zone.set_facecolor((1.0, 0.82, 0.0, 0.35))
                safety_zone.set_edgecolor((1.0, 0.45, 0.0, 0.95))
                safety_zone.set_linewidth(2.6)
            else:
                safety_zone.set_facecolor((1.0, 0.85, 0.0, 0.10))
                safety_zone.set_edgecolor((1.0, 0.62, 0.0, 0.35))
                safety_zone.set_linewidth(1.4)

        if cbf_active and critical_obstacle is not None:
            obstacle = obstacles[critical_obstacle]
            obstacle_name = getattr(
                obstacle,
                "name",
                f"obstacle {critical_obstacle + 1}",
            )
            cbf_text.set_text(f"CBF active: {obstacle_name}")
            cbf_text.set_color("tab:orange")
        elif cbf_history_data is not None:
            cbf_text.set_text("CBF inactive")
            cbf_text.set_color("0.25")
        else:
            cbf_text.set_text("")

        if state.shape[1] >= 7:
            time_text.set_text(
                f"t = {t[index]:.1f} s, sigma = {reference_time[index]:.1f} s"
            )
        else:
            time_text.set_text(f"t = {t[index]:.1f} s")

        return (
            *cbf_safety_patches,
            actual_path,
            reference_point,
            actual_boat,
            actual_front,
            cbf_text,
            time_text,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )

    available_writers = animation.writers.list()
    if output_path.suffix.lower() == ".mp4" and "ffmpeg" in available_writers:
        writer = animation.FFMpegWriter(fps=fps)
    elif output_path.suffix.lower() == ".gif" and "pillow" in available_writers:
        writer = animation.PillowWriter(fps=fps)
    elif "pillow" in available_writers:
        output_path = output_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    else:
        raise RuntimeError("No animation writer available. Install ffmpeg or pillow.")

    anim.save(output_path, writer=writer)
    plt.close(fig)

    return output_path


def plot_results(
    t,
    state,
    controller=None,
    controller_history=None,
    Tu_limits=None,
    Tv_limits=None,
    Tr_limits=None,
    show_reference=True,
    disturbance_info=None,
    disturbance_fun=None,
    obstacles=None,
    cbf_history_data=None,
    governor_history_data=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    save_dir=None,
    show=False,
):
    x = state[:, 0]
    y = state[:, 1]
    psi = state[:, 2]
    u = state[:, 3]
    v = state[:, 4]
    r = state[:, 5]
    p_s, _ = lookahead_point_from_state(state, lookahead_L)
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    ax.plot(x, y, color="tab:blue", label="USV path", linewidth=1.8, zorder=2)

    if show_reference:
        ref_curve_duration = trajectory_reference_curve_duration()
        t_one_lap = np.linspace(0.0, ref_curve_duration, 700)
        x_ref, y_ref, _, _, _, _ = figure_eight_reference(t_one_lap)
        x_ref_progress, y_ref_progress, _, _, _, _ = figure_eight_reference(
            reference_time
        )
        ax.plot(
            x_ref,
            y_ref,
            linestyle="--",
            dashes=(4, 4),
            color="0.20",
            alpha=0.90,
            linewidth=1.3,
            label=trajectory_label(),
            zorder=3,
        )
        ax.scatter(
            x_ref_progress[-1],
            y_ref_progress[-1],
            color="#00C853",
            edgecolors="black",
            linewidths=0.5,
            s=16,
            label="reference point",
            zorder=6,
        )

    if obstacles:
        for index, obstacle in enumerate(obstacles):
            label = "CBF safety zone" if index == 0 else "_nolegend_"
            safety_zone = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius + CBF_MARGIN,
                edgecolor=(1.0, 0.55, 0.0, 0.70),
                facecolor=(1.0, 0.85, 0.0, 0.12),
                linestyle="--",
                linewidth=1.3,
                label=label,
                zorder=1,
            )
            ax.add_patch(safety_zone)
        draw_obstacles(ax, obstacles)

    boat_length = VESSEL_SIDE_M
    boat_width = VESSEL_SIDE_M
    sample_count = min(10, len(x))
    active_traj = get_active_trajectory()
    if active_traj == "figure_eight":
        one_lap_end = max(1, len(x) // 2)
    else:
        one_lap_end = len(x)
    sample_indices = np.linspace(0, one_lap_end, sample_count, endpoint=False, dtype=int)

    for index in sample_indices[1:]:
        draw_boat(
            ax,
            x[index],
            y[index],
            psi[index],
            length=boat_length,
            width=boat_width,
            color="#1E293B",
            front_color="#FF2A00",
            alpha=0.65,
            linewidth=1.0,
            zorder=4,
        )

    draw_boat(
        ax,
        x[-1],
        y[-1],
        psi[-1],
        length=boat_length,
        width=boat_width,
        color="#111827",
        front_color="#FF2A00",
        alpha=0.92,
        linewidth=1.5,
        zorder=5,
    )

    ax.axis("equal")
    x_pad = 0.08 * max(np.ptp(x), 1.0)
    y_pad_top = 0.10 * max(np.ptp(y), 1.0)
    y_pad_bottom = 0.22 * max(np.ptp(y), 1.0)
    ax.set_xlim(np.min(x) - x_pad, np.max(x) + x_pad)
    ax.set_ylim(np.min(y) - y_pad_bottom, np.max(y) + y_pad_top)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("PlaDyPos Omnidirectional USV Trajectory with Heading")
    ax.grid(True)

    if disturbance_info is None and disturbance_fun is not None:
        disturbance_info = getattr(disturbance_fun, "info", None)

    if disturbance_info is not None:
        draw_disturbance_arrow(
            ax,
            magnitude=disturbance_info["magnitude"],
            direction_deg=disturbance_info["direction_deg"],
        )

    ax.legend(loc="upper right", fontsize=8)

    trajectory_fig = fig

    orientation_fig, ax_orientation = plt.subplots()
    ax_orientation.plot(t, np.rad2deg(psi))
    ax_orientation.set_xlabel("t [s]")
    ax_orientation.set_ylabel(r"$\psi$ [deg]")
    ax_orientation.set_title(r"Heading Angle $\psi$")
    ax_orientation.grid(True)

    velocities_fig, ax_velocities = plt.subplots()
    ax_velocities.plot(t, u, label="u (surge)")
    ax_velocities.plot(t, v, label="v (sway)")
    ax_velocities.plot(t, r, label="r (yaw rate)")
    ax_velocities.set_xlabel("t [s]")
    ax_velocities.set_ylabel("body-frame velocities")
    ax_velocities.set_title("Body-Frame Velocities (3-DOF)")
    ax_velocities.legend()
    ax_velocities.grid(True)

    figures = {
        "trajectory": trajectory_fig,
        "tracking_error": plot_tracking_error(
            t,
            state,
            lookahead_L,
            governor_history_data=governor_history_data,
        ),
        "orientation": orientation_fig,
        "velocities": velocities_fig,
    }

    if governor_history_data is not None:
        figures["reference_governor"] = plot_reference_governor(
            t,
            governor_history_data,
        )

    if disturbance_fun is not None:
        figures["disturbances"] = plot_disturbance(t, state, disturbance_fun)

    if controller is not None:
        tau_required = None
        tau_delivered = None
        thruster_forces = None
        if controller_history is not None:
            tau_required, tau_delivered, thruster_forces = controller_history

        figures["torque_saturation"] = plot_torque_saturation(
            t,
            state,
            controller,
            Tu_limits=Tu_limits,
            Tv_limits=Tv_limits,
            Tr_limits=Tr_limits,
            tau_required=tau_required,
            tau_delivered=tau_delivered,
        )

        if hasattr(controller, "thruster_forces"):
            figures["thruster_forces"] = plot_thruster_forces(
                t,
                state,
                controller,
                forces=thruster_forces,
            )

    if cbf_history_data is not None:
        figures["cbf_barrier"] = plot_cbf_barrier(t, cbf_history_data)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")

    return figures
