"""MuJoCo elevation map: absolute ground_z + ROS 2 GridMap / TF + viewer markers.

TF ``parent`` → ``elevation_map`` (default ``world`` → ``elevation_map``):
  - XY / yaw from ``torso_link``; pitch/roll = 0; Z = 0 (world origin).
Layer ``elevation``: absolute world-frame ``ground_z`` (miss → 0.0).
"""

from __future__ import annotations

import numpy as np

import mujoco

N_FORWARD_CELLS = 15
N_LATERAL_CELLS = 15
CELL_RESOLUTION = 0.1
TOTAL_CELLS = N_FORWARD_CELLS * N_LATERAL_CELLS

FORWARD_HALF_SPAN = (N_FORWARD_CELLS - 1) / 2 * CELL_RESOLUTION
LATERAL_HALF_SPAN = (N_LATERAL_CELLS - 1) / 2 * CELL_RESOLUTION
RAY_ALTITUDE_ABOVE_SENSOR = 3.0

MARKER_RADIUS = 0.025
MARKER_RGBA = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)
_RAY_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def _build_geomgroup_mask(terrain_geom_group: int) -> np.ndarray:
    if terrain_geom_group < 0 or terrain_geom_group >= mujoco.mjNGROUP:
        raise ValueError(
            f"terrain_geom_group must be in [0, {mujoco.mjNGROUP - 1}], "
            f"got {terrain_geom_group}."
        )
    geomgroup = np.zeros(mujoco.mjNGROUP, dtype=np.ubyte)
    geomgroup[terrain_geom_group] = 1
    return geomgroup


def _build_grid_local_offsets() -> np.ndarray:
    """Local offsets in grid_map ColMajor order: index = ix + iy * n_forward.

    ``(ix, iy) = (0, 0)`` is front-left (+X_local, +Y_local).
    """
    # Front → rear along +X; left → right along +Y (decreasing lateral).
    forward = np.linspace(FORWARD_HALF_SPAN, -FORWARD_HALF_SPAN, N_FORWARD_CELLS)
    lateral = np.linspace(LATERAL_HALF_SPAN, -LATERAL_HALF_SPAN, N_LATERAL_CELLS)
    grid_forward, grid_lateral = np.meshgrid(forward, lateral, indexing="ij")
    return np.stack(
        [
            grid_forward.ravel(order="F"),
            grid_lateral.ravel(order="F"),
            np.zeros(TOTAL_CELLS, dtype=np.float64),
        ],
        axis=-1,
    )


_GRID_LOCAL_OFFSETS = _build_grid_local_offsets()


class ElevationMapPublisher:
    """Publish absolute ``ground_z`` GridMap and draw red hit markers in the viewer."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        topic: str,
        sensor_body_name: str = "torso_link",
        terrain_geom_group: int = 2,
        node_name: str = "gear_sonic_elevation_map",
        qos_depth: int = 10,
        parent_frame_id: str = "world",
        grid_frame_id: str = "elevation_map",
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

        self._hit_geom_id = np.array([-1], dtype=np.int32)
        self._elevation_heights = np.zeros(TOTAL_CELLS, dtype=np.float64)
        self._hit_positions_xyz = np.zeros((TOTAL_CELLS, 3), dtype=np.float64)
        self._hit_positions_valid = np.zeros(TOTAL_CELLS, dtype=bool)
        self._ray_origins = np.empty((TOTAL_CELLS, 3), dtype=np.float64)
        self._last_hit_positions: list[np.ndarray] = []

        try:
            import rclpy
            import tf2_ros
            from geometry_msgs.msg import TransformStamped
            from grid_map_msgs.msg import GridMap
            from rclpy.node import Node
            from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages (rclpy, grid_map_msgs, tf2_ros) required for "
                "--enable-ros2-elevation-map."
            ) from exc

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()

        self._parent_frame_id = parent_frame_id
        self._grid_frame_id = grid_frame_id
        self._rclpy = rclpy
        self._node = Node(node_name)
        self._pub = self._node.create_publisher(GridMap, topic, qos_depth)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)
        self._tf_message = TransformStamped()
        self._torso_tf_message = TransformStamped()
        self._sensor_body_name = sensor_body_name

        layout = MultiArrayLayout()
        dim_col = MultiArrayDimension()
        dim_col.label = "column_index"
        dim_col.size = N_LATERAL_CELLS
        dim_col.stride = TOTAL_CELLS
        dim_row = MultiArrayDimension()
        dim_row.label = "row_index"
        dim_row.size = N_FORWARD_CELLS
        dim_row.stride = N_FORWARD_CELLS
        layout.dim = [dim_col, dim_row]
        layout.data_offset = 0

        layer = Float32MultiArray()
        layer.layout = layout
        layer.data = [0.0] * TOTAL_CELLS

        self._grid_map_msg = GridMap()
        self._grid_map_msg.header.frame_id = grid_frame_id
        self._grid_map_msg.info.resolution = float(CELL_RESOLUTION)
        self._grid_map_msg.info.length_x = float(N_FORWARD_CELLS * CELL_RESOLUTION)
        self._grid_map_msg.info.length_y = float(N_LATERAL_CELLS * CELL_RESOLUTION)
        self._grid_map_msg.info.pose.orientation.w = 1.0
        self._grid_map_msg.layers = ["elevation"]
        self._grid_map_msg.basic_layers = ["elevation"]
        self._grid_map_msg.outer_start_index = 0
        self._grid_map_msg.inner_start_index = 0
        self._grid_map_msg.data = [layer]

        print(
            f"[ElevationMap] ground_z on {topic!r}, frame {grid_frame_id!r}, "
            f"TF {parent_frame_id!r}→{grid_frame_id!r} (torso XY, z=0, yaw-only) "
            f"and {parent_frame_id!r}→{sensor_body_name!r} (full pose for sensor_z)."
        )

    def publish(self, mj_data: mujoco.MjData) -> None:
        elevation_heights, hit_positions, sensor_position, yaw = (
            self._compute_elevation_map(mj_data)
        )
        self._last_hit_positions = hit_positions

        stamp = self._node.get_clock().now().to_msg()
        qx, qy, qz, qw = _yaw_to_quat_xyzw(yaw)

        # Level elevation_map frame: torso XY, world Z = 0, yaw-only.
        tf_msg = self._tf_message
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self._parent_frame_id
        tf_msg.child_frame_id = self._grid_frame_id
        tf_msg.transform.translation.x = float(sensor_position[0])
        tf_msg.transform.translation.y = float(sensor_position[1])
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        # Full torso pose so the controller can TF-lookup sensor_z.
        quat_wxyz = mj_data.xquat[self._sensor_body_id]
        torso_tf = self._torso_tf_message
        torso_tf.header.stamp = stamp
        torso_tf.header.frame_id = self._parent_frame_id
        torso_tf.child_frame_id = self._sensor_body_name
        torso_tf.transform.translation.x = float(sensor_position[0])
        torso_tf.transform.translation.y = float(sensor_position[1])
        torso_tf.transform.translation.z = float(sensor_position[2])
        torso_tf.transform.rotation.x = float(quat_wxyz[1])
        torso_tf.transform.rotation.y = float(quat_wxyz[2])
        torso_tf.transform.rotation.z = float(quat_wxyz[3])
        torso_tf.transform.rotation.w = float(quat_wxyz[0])
        self._tf_broadcaster.sendTransform([tf_msg, torso_tf])

        msg = self._grid_map_msg
        msg.header.stamp = stamp
        msg.header.frame_id = self._grid_frame_id
        msg.data[0].data = elevation_heights.astype(np.float32, copy=False).tolist()
        self._pub.publish(msg)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def update_viewer_markers(
        self,
        viewer: mujoco.viewer.Handle,
        mj_data: mujoco.MjData | None = None,
    ) -> None:
        if not self._last_hit_positions:
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

    def _cast_ray_distance(self, mj_data: mujoco.MjData, origin: np.ndarray) -> float:
        return float(
            mujoco.mj_ray(
                self._mj_model,
                mj_data,
                origin,
                _RAY_DIRECTION,
                self._terrain_geomgroup,
                1,
                -1,
                self._hit_geom_id,
            )
        )

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
        world_offsets = _GRID_LOCAL_OFFSETS @ _yaw_rotation_matrix(yaw).T

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

        for cell_index in range(TOTAL_CELLS):
            distance = self._cast_ray_distance(mj_data, ray_origins[cell_index])
            if distance < 0:
                continue
            ground_z = ray_origin_z - distance
            elevation_heights[cell_index] = ground_z
            hit_positions_xyz[cell_index, 0] = ray_origins[cell_index, 0]
            hit_positions_xyz[cell_index, 1] = ray_origins[cell_index, 1]
            hit_positions_xyz[cell_index, 2] = ground_z
            hit_positions_valid[cell_index] = True

        hit_positions: list[np.ndarray] = []
        if hit_positions_valid.any():
            hit_positions = list(hit_positions_xyz[hit_positions_valid])

        return elevation_heights, hit_positions, sensor_position, yaw
