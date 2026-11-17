from dataclasses import dataclass


@dataclass(frozen=True)
class CircularObstacle:
    """Circular obstacle represented in the inertial x-y plane."""

    x: float
    y: float
    radius: float
    name: str = ""


OBSTACLES = (
    CircularObstacle(x=10.0, y=-6.5, radius=0.75, name="buoy_1"),
    CircularObstacle(x=-13.0, y=-7.5, radius=0.25, name="buoy_2"),
)

OBSTACLES_SPIRAL = (
    CircularObstacle(x=-4.29, y=-4.29, radius=0.50, name="buoy_1"),
    CircularObstacle(x=4.38, y=8.59, radius=0.65, name="buoy_2"),
)

OBSTACLES_CLOVER = (
    CircularObstacle(x=-2.55, y=9.95, radius=0.65, name="buoy_1"),
    CircularObstacle(x=-7.50, y=-7.50, radius=0.45, name="buoy_2"),
)

OBSTACLES_BY_TRAJECTORY = {
    "figure_eight": OBSTACLES,
    "spiral": OBSTACLES_SPIRAL,
    "clover": OBSTACLES_CLOVER,
}


def configured_obstacles(enabled=True, obstacles=None, trajectory_type="figure_eight"):
    """Returns the configured obstacle set for the given trajectory, or () if disabled."""
    if not enabled:
        return ()

    if obstacles is not None:
        return tuple(obstacles)

    key = str(trajectory_type).strip().lower()
    return tuple(OBSTACLES_BY_TRAJECTORY.get(key, OBSTACLES))
