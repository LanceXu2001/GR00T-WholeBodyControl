"""Publish sensor_msgs/PointCloud2 from MuJoCo LiDAR raycasting (optional ROS 2)."""

from __future__ import annotations

import time

import numpy as np
import mujoco

_RAY_STEP_EPSILON = 1e-4
_RAY_MAX_STEPS = 64
_RAY_START_OFFSET = 0.15
_MIN_ENVIRONMENT_HIT_DISTANCE = 0.10


class LidarPointCloudPublisher:
    """Simulate a LiDAR sensor attached to a MuJoCo site and publish PointCloud2."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        topic: str,
        site_name: str = "lidar",
        exclude_robot_root_body_name: str = "pelvis",
        batch_bodyexclude_name: str = "torso_link",
        scan_type: str = "mid360",
        backend: str = "jax",
        node_name: str = "gear_sonic_lidar",
        cutoff_distance: float = 100.0,
    ) -> None:
        from mujoco_lidar import MjLidarWrapper, scan_gen
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2, PointField

        self._mj_model = mj_model
        self._cutoff_distance = cutoff_distance
        self._site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._scan_type = scan_type
        self._backend_name = backend
        self._frame_count = 0

        root_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, exclude_robot_root_body_name
        )
        excluded_body_ids = self._collect_subtree_body_ids(mj_model, root_body_id)
        self._excluded_geom_ids = frozenset(
            geom_id
            for geom_id in range(mj_model.ngeom)
            if mj_model.geom_bodyid[geom_id] in excluded_body_ids
        )

        geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.ubyte)
        geomgroup[3:] = 0

        if scan_type == "mid360":
            batch_exclude_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, batch_bodyexclude_name
            )
            self._livox_generator = scan_gen.LivoxGenerator("mid360")
            self._rays_theta, self._rays_phi = self._livox_generator.sample_ray_angles()
            self._use_batch_backend = True
            self._lidar = MjLidarWrapper(
                mj_model,
                site_name=site_name,
                backend=backend,
                cutoff_dist=cutoff_distance,
                args={"bodyexclude": batch_exclude_id, "geomgroup": geomgroup},
            )
        elif scan_type == "grid":
            self._use_batch_backend = False
            self._livox_generator = None
            rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(
                num_ray_cols=60,
                num_ray_rows=16,
                phi_range=(-np.pi / 2 + 0.05, np.pi / 3),
            )
            self._rays_theta = np.ascontiguousarray(rays_theta, dtype=np.float64)
            self._rays_phi = np.ascontiguousarray(rays_phi, dtype=np.float64)
            self._lidar = None
        else:
            raise ValueError(f"Unsupported scan_type: {scan_type!r} (use 'mid360' or 'grid').")

        self._hit_geom_buffer = np.array([-1], dtype=np.int32)

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node(node_name)
        self._pub = self._node.create_publisher(PointCloud2, topic, 1)
        self._PointCloud2 = PointCloud2
        self._fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        compute_device = "GPU (JAX/CUDA)" if backend == "jax" else (
            "GPU (Taichi)" if backend == "taichi" else "CPU"
        )
        if scan_type == "mid360" and backend == "cpu":
            compute_device = "CPU (mj_multiRay batch, ~24k rays/frame)"
        print(
            f"[LidarPointCloud] scan={scan_type!r}, backend={backend!r} ({compute_device}), "
            f"rays={len(self._rays_theta)}, site={site_name!r}, "
            f"batch bodyexclude={batch_bodyexclude_name!r}, "
            f"grid geom filter: {len(self._excluded_geom_ids)} geoms under {exclude_robot_root_body_name!r}."
        )

    @staticmethod
    def _collect_subtree_body_ids(
        mj_model: mujoco.MjModel, root_body_id: int
    ) -> frozenset[int]:
        collected: set[int] = set()
        stack = [root_body_id]
        while stack:
            body_id = stack.pop()
            collected.add(body_id)
            for child_id in range(mj_model.nbody):
                if mj_model.body_parentid[child_id] == body_id:
                    stack.append(child_id)
        return frozenset(collected)

    def _ray_directions_world(self, mj_data: mujoco.MjData) -> np.ndarray:
        site_rotation = mj_data.site_xmat[self._site_id].reshape(3, 3)
        local_directions = np.stack(
            (
                np.cos(self._rays_phi) * np.cos(self._rays_theta),
                np.cos(self._rays_phi) * np.sin(self._rays_theta),
                np.sin(self._rays_phi),
            ),
            axis=-1,
        )
        world_directions = local_directions @ site_rotation.T
        return world_directions / np.linalg.norm(
            world_directions, axis=1, keepdims=True
        )

    def _ray_hit_environment(
        self,
        mj_data: mujoco.MjData,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> np.ndarray | None:
        excluded = self._excluded_geom_ids
        current_origin = ray_origin.copy()
        traveled = 0.0

        for _ in range(_RAY_MAX_STEPS):
            if traveled >= self._cutoff_distance:
                return None
            distance = mujoco.mj_ray(
                self._mj_model,
                mj_data,
                current_origin,
                ray_direction,
                None,
                1,
                -1,
                self._hit_geom_buffer,
            )
            if distance < 0:
                return None
            geom_id = int(self._hit_geom_buffer[0])
            if geom_id not in excluded:
                return current_origin + ray_direction * distance
            step = distance + _RAY_STEP_EPSILON
            traveled += step
            current_origin = current_origin + ray_direction * step

        return None

    def _cast_grid_world_points(self, mj_data: mujoco.MjData) -> np.ndarray:
        site_position = mj_data.site_xpos[self._site_id]
        world_directions = self._ray_directions_world(mj_data)
        hit_points: list[np.ndarray] = []

        for direction in world_directions:
            ray_origin = site_position + direction * _RAY_START_OFFSET
            hit_position = self._ray_hit_environment(mj_data, ray_origin, direction)
            if hit_position is not None:
                hit_points.append(hit_position)

        if not hit_points:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(hit_points)

    def _cast_mid360_world_points(self, mj_data: mujoco.MjData) -> np.ndarray:
        self._rays_theta, self._rays_phi = self._livox_generator.sample_ray_angles()
        self._rays_theta = np.ascontiguousarray(self._rays_theta, dtype=np.float64)
        self._rays_phi = np.ascontiguousarray(self._rays_phi, dtype=np.float64)

        self._lidar.trace_rays(mj_data, self._rays_theta, self._rays_phi)
        distances = np.asarray(self._lidar.get_distances(), dtype=np.float64)
        points_sensor = np.asarray(self._lidar.get_hit_points(), dtype=np.float64)

        site_position = mj_data.site_xpos[self._site_id]
        site_rotation = mj_data.site_xmat[self._site_id].reshape(3, 3)
        points_world = points_sensor @ site_rotation.T + site_position

        valid = distances > _MIN_ENVIRONMENT_HIT_DISTANCE
        return points_world[valid]

    def _cast_world_points(self, mj_data: mujoco.MjData) -> np.ndarray:
        if self._use_batch_backend:
            return self._cast_mid360_world_points(mj_data)
        return self._cast_grid_world_points(mj_data)

    def publish(self, mj_data: mujoco.MjData) -> None:
        start_time = time.perf_counter()
        points_world = self._cast_world_points(mj_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        self._frame_count += 1
        if self._frame_count == 1 or self._frame_count % 50 == 0:
            print(
                f"[LidarPointCloud] frame {self._frame_count}: {len(points_world)} points, "
                f"{elapsed_ms:.1f} ms ({self._scan_type}/{self._backend_name})."
            )

        if len(points_world) == 0:
            print("[LidarPointCloud] Warning: no valid environment points in this frame.")

        msg = self._PointCloud2()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.fields = self._fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.height = 1
        msg.is_dense = True
        msg.width = len(points_world)
        msg.row_step = 12 * len(points_world)
        msg.data = points_world.astype(np.float32).tobytes()
        self._pub.publish(msg)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()
