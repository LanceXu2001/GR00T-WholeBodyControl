"""Elevation-map computation, viewer visualization, and ROS 2 publishing.

Pipeline per call to ``publish()``:
1. Read robot sensor-body pose from ``mjData`` (default: ``torso_link``) and extract
   **heading** (yaw only, ignoring roll/pitch) to build a heading-aligned local frame.
2. Cast one downward ray per grid cell using a MuJoCo ``geomgroup`` mask so only
   terrain geoms are hit — one ``mj_ray`` per cell.  Cells with no terrain hit use 0.0 m.
3. Optionally draw red sphere markers in the passive viewer; ``update_viewer_markers()``
   reuses the hit points cached by ``publish()`` (no second ``mj_ray`` pass).
4. Publish a ``grid_map_msgs/GridMap`` with a single ``elevation`` layer.

Grid layout (viewed from above, heading = +X):
    n_forward_cells = 15  → along heading (+X_local), meshgrid row index
    n_lateral_cells = 15  → perpendicular (+Y_local), meshgrid column index
Flat cell order (MuJoCo + ROS): F-order of ``height[forward, lateral]`` —
    cell 0 = (-0.7, -0.7), cell 1 = one step forward, then next lateral column.
ROS uses ``gridmap_column`` layout (same F-order flat).
``header.frame_id`` defaults to ``torso_link`` with identity pose so the map rotates
with the robot (RViz ignores ``pose.orientation`` when frame is ``world``).
"""

from __future__ import annotations

import numpy as np

import mujoco


# ── grid constants ────────────────────────────────────────────────────────────
N_FORWARD_CELLS = 15   # number of cells along robot heading direction
N_LATERAL_CELLS = 15   # number of cells perpendicular to heading
CELL_RESOLUTION = 0.1  # meters per cell
TOTAL_CELLS = N_FORWARD_CELLS * N_LATERAL_CELLS  # 225

# Half-spans (metres from robot centre to grid edge)
FORWARD_HALF_SPAN = (N_FORWARD_CELLS - 1) / 2 * CELL_RESOLUTION   # 0.8 m
LATERAL_HALF_SPAN = (N_LATERAL_CELLS - 1) / 2 * CELL_RESOLUTION   # 0.5 m

# Ray origin altitude above the sensor body (clears the robot body)
RAY_ALTITUDE_ABOVE_SENSOR = 3.0  # metres

# Visualisation sphere size
MARKER_RADIUS = 0.025  # metres
MARKER_RGBA   = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)


_RAY_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Extract yaw (rad) from a MuJoCo ``(w, x, y, z)`` quaternion."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for rotation about +Z by *yaw*."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _build_geomgroup_mask(terrain_geom_group: int) -> np.ndarray:
    """Return a MuJoCo geomgroup mask that includes only *terrain_geom_group*."""
    if terrain_geom_group < 0 or terrain_geom_group >= mujoco.mjNGROUP:
        raise ValueError(
            f"terrain_geom_group must be in [0, {mujoco.mjNGROUP - 1}], "
            f"got {terrain_geom_group}."
        )
    geomgroup = np.zeros(mujoco.mjNGROUP, dtype=np.ubyte)
    geomgroup[terrain_geom_group] = 1
    return geomgroup


def _build_grid_local_offsets() -> np.ndarray:
    """Return (TOTAL_CELLS, 3) local offset array in F-order of ``[forward, lateral]``.

    Flat index ``k = forward + lateral * N_FORWARD_CELLS`` so cell 0 is the
    rear-right corner ``(-half, -half)`` and cell 1 is one step forward.
    Z_local is 0 for all cells (flat horizontal grid).
    """
    forward_offsets = np.linspace(-FORWARD_HALF_SPAN, FORWARD_HALF_SPAN, N_FORWARD_CELLS)
    lateral_offsets = np.linspace(-LATERAL_HALF_SPAN, LATERAL_HALF_SPAN, N_LATERAL_CELLS)

    # Shape: (N_FORWARD_CELLS, N_LATERAL_CELLS)
    grid_forward, grid_lateral = np.meshgrid(forward_offsets, lateral_offsets, indexing="ij")
    grid_zeros = np.zeros_like(grid_forward)

    # F-order: forward varies fastest → cell 1 is ahead of cell 0, not to the side.
    return np.stack(
        [
            grid_forward.ravel(order="F"),
            grid_lateral.ravel(order="F"),
            grid_zeros.ravel(order="F"),
        ],
        axis=-1,
    )


# Pre-compute local offsets once (they never change)
_GRID_LOCAL_OFFSETS = _build_grid_local_offsets()


class ElevationMapPublisher:
    """Compute and publish a heading-aligned elevation map each simulation step."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        topic: str,
        sensor_body_name: str = "torso_link",
        terrain_geom_group: int = 2,
        node_name: str = "gear_sonic_elevation_map",
        qos_depth: int = 10,
        map_frame_id: str | None = None,
    ) -> None:
        self._mj_model = mj_model
        self._sensor_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, sensor_body_name
        )
        if self._sensor_body_id < 0:
            raise ValueError(
                f"[ElevationMap] Sensor body '{sensor_body_name}' not found in MuJoCo model."
            )
        self._terrain_geomgroup = _build_geomgroup_mask(terrain_geom_group)
        print(
            f"[ElevationMap] Using geomgroup filter: terrain only (group={terrain_geom_group})."
        )

        self._hit_geom_id = np.array([-1], dtype=np.int32)
        self._elevation_heights = np.zeros(TOTAL_CELLS, dtype=np.float64)
        self._hit_positions_xyz = np.zeros((TOTAL_CELLS, 3), dtype=np.float64)
        self._hit_positions_valid = np.zeros(TOTAL_CELLS, dtype=bool)
        self._ray_origins = np.empty((TOTAL_CELLS, 3), dtype=np.float64)
        self._last_hit_positions: list[np.ndarray] = []

        try:
            import rclpy
            from grid_map_msgs.msg import GridMap
            from rclpy.node import Node
            from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
        except ImportError as exc:
            raise ImportError(
                "ROS 2 Python packages (rclpy, std_msgs, grid_map_msgs) "
                "not found. Source ROS 2 (e.g. /opt/ros/humble/setup.bash) so that "
                "`ros-humble-grid-map` is on PYTHONPATH, then re-run with "
                "--enable-ros2-elevation-map."
            ) from exc

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()

        self._rclpy = rclpy
        self._node = Node(node_name)
        self._pub = self._node.create_publisher(GridMap, topic, qos_depth)
        self._map_frame_id = map_frame_id or sensor_body_name

        # grid_map_rviz_plugin requires gridmap_column layout (ColMajor Eigen).
        # Flat buffer is already F-order of height[forward, lateral] (matches MuJoCo).
        elevation_layout = MultiArrayLayout()
        dim_col = MultiArrayDimension()
        dim_col.label = "column_index"
        dim_col.size = N_LATERAL_CELLS
        dim_col.stride = N_FORWARD_CELLS * N_LATERAL_CELLS
        dim_row = MultiArrayDimension()
        dim_row.label = "row_index"
        dim_row.size = N_FORWARD_CELLS
        dim_row.stride = N_FORWARD_CELLS
        elevation_layout.dim = [dim_col, dim_row]
        elevation_layout.data_offset = 0

        self._elevation_data = [0.0] * TOTAL_CELLS
        elevation_layer = Float32MultiArray()
        elevation_layer.layout = elevation_layout
        elevation_layer.data = self._elevation_data
        self._grid_map_msg = GridMap()
        self._grid_map_msg.header.frame_id = self._map_frame_id
        self._grid_map_msg.info.resolution = float(CELL_RESOLUTION)
        self._grid_map_msg.info.length_x = float(N_FORWARD_CELLS  * CELL_RESOLUTION)
        self._grid_map_msg.info.length_y = float(N_LATERAL_CELLS  * CELL_RESOLUTION)
        self._grid_map_msg.info.pose.orientation.w = 1.0
        self._grid_map_msg.layers = ["elevation"]
        self._grid_map_msg.basic_layers = ["elevation"]
        self._grid_map_msg.outer_start_index = 0
        self._grid_map_msg.inner_start_index = 0
        self._grid_map_msg.data = [elevation_layer]

    def publish(self, mj_data: mujoco.MjData) -> None:
        """Cast rays, cache hit points for debugging, and publish ROS 2 message."""
        elevation_heights, hit_positions, _sensor_position, _yaw = self._compute_elevation_map(
            mj_data
        )
        self._last_hit_positions = hit_positions

        msg = self._grid_map_msg
        msg.header.stamp = self._node.get_clock().now().to_msg()
        # Data is heading-aligned in header.frame_id; identity pose at frame origin.
        msg.info.pose.position.x = 0.0
        msg.info.pose.position.y = 0.0
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.x = 0.0
        msg.info.pose.orientation.y = 0.0
        msg.info.pose.orientation.z = 0.0
        msg.info.pose.orientation.w = 1.0

        # elevation_heights is already F-order of height[forward, lateral] — same as before.
        msg.data[0].data = elevation_heights.astype(np.float32, copy=False).tolist()
        self._pub.publish(msg)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def update_viewer_markers(
        self,
        viewer: mujoco.viewer.Handle,
        mj_data: mujoco.MjData | None = None,
        *,
        recompute: bool = False,
    ) -> None:
        if recompute or not self._last_hit_positions:
            if mj_data is None:
                return
            _, hit_positions, _, _ = self._compute_elevation_map(mj_data)
            self._last_hit_positions = hit_positions
        self._draw_hit_markers(viewer, self._last_hit_positions)

    def _draw_hit_markers(
        self, viewer: mujoco.viewer.Handle, hit_positions: list[np.ndarray]
    ) -> None:
        max_geoms = viewer.user_scn.maxgeom
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            for hit_position in hit_positions[:max_geoms]:
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([MARKER_RADIUS, 0.0, 0.0]),
                    pos=hit_position,
                    mat=np.eye(3).flatten().astype(np.float64),
                    rgba=MARKER_RGBA,
                )
                viewer.user_scn.ngeom += 1

    def _cast_ray_distance(
        self,
        mj_data: mujoco.MjData,
        origin: np.ndarray,
    ) -> float:
        dist = mujoco.mj_ray(
            self._mj_model,
            mj_data,
            origin,
            _RAY_DIRECTION,
            self._terrain_geomgroup,
            1,
            -1,
            self._hit_geom_id,
        )
        return float(dist)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()

    def _compute_elevation_map(
        self,
        mj_data: mujoco.MjData,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
        sensor_position = np.asarray(
            mj_data.xpos[self._sensor_body_id], dtype=np.float64
        ).reshape(3)
        yaw = _yaw_from_quat_wxyz(mj_data.xquat[self._sensor_body_id])
        rotation_local_to_world = _yaw_rotation_matrix(yaw)

        world_offsets = _GRID_LOCAL_OFFSETS @ rotation_local_to_world.T
        ray_origin_z = sensor_position[2] + RAY_ALTITUDE_ABOVE_SENSOR
        ray_origins = self._ray_origins
        ray_origins[:, 0] = sensor_position[0] + world_offsets[:, 0]
        ray_origins[:, 1] = sensor_position[1] + world_offsets[:, 1]
        ray_origins[:, 2] = ray_origin_z

        elevation_heights = self._elevation_heights
        elevation_heights.fill(0.0)
        hit_positions_xyz = self._hit_positions_xyz
        hit_positions_valid = self._hit_positions_valid
        hit_positions_valid.fill(False)
        hit_positions: list[np.ndarray] = []

        for cell_index in range(TOTAL_CELLS):
            distance = self._cast_ray_distance(mj_data, ray_origins[cell_index])
            if distance < 0:
                continue
            ground_z = ray_origin_z - distance
            elevation_heights[cell_index] = sensor_position[2] - ground_z - 0.5
            hit_positions_xyz[cell_index, 0] = ray_origins[cell_index, 0]
            hit_positions_xyz[cell_index, 1] = ray_origins[cell_index, 1]
            hit_positions_xyz[cell_index, 2] = ground_z
            hit_positions_valid[cell_index] = True

        if hit_positions_valid.any():
            hit_positions = list(hit_positions_xyz[hit_positions_valid])

        return elevation_heights, hit_positions, sensor_position, yaw
