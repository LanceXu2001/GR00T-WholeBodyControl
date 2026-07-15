"""ROS2 subscriber node that records robot state and sim body state to CSV per trial."""

import csv
import json
import threading
from pathlib import Path

import msgpack
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import ByteMultiArray, Empty, Float64MultiArray, String


class Ros2Collector:
    """Subscribes to four ROS2 topics and writes CSV files inside each trial directory.

    Thread model:
        - ROS2 callbacks execute on the spin thread (call spin_forever() in a daemon thread).
        - start_trial() / stop_trial() are called from the main thread.
        - A lock guards CSV file access between the two threads.

    Topics:
        /sim/bodies/names      (latched)  → body name list, used for CSV column headers
        G1Env/env_state_act    (50 Hz)    → robot/robot_state.csv
        /sim/bodies/poses      (≤200 Hz)  → sim/body_pos.csv, sim/body_quat.csv
        /sim/bodies/velocities (≤200 Hz)  → sim/body_lin_vel.csv, sim/body_ang_vel.csv

    Fall detection:
        pelvis_z is updated each time /sim/bodies/poses arrives.
        The main trial loop reads this value to detect falls.
    """

    def __init__(self) -> None:
        rclpy.init()
        self._node = Node("motion_collect_ros2_collector")

        self.body_names: list = []
        self.pelvis_z: float = float("inf")

        self._recording = False
        self._lock = threading.Lock()
        self._csv_files: dict = {}
        self._csv_writers: dict = {}
        self._body_names_ready = threading.Event()

        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._node.create_subscription(String, "/sim/bodies/names", self._on_body_names, latched_qos)
        self._node.create_subscription(ByteMultiArray, "G1Env/env_state_act", self._on_robot_state, 10)
        self._node.create_subscription(PoseArray, "/sim/bodies/poses", self._on_body_poses, 10)
        self._node.create_subscription(
            Float64MultiArray, "/sim/bodies/velocities", self._on_body_velocities, 10
        )
        self._sim_reset_publisher = self._node.create_publisher(Empty, "/sim/reset", 10)
        self._release_band_publisher = self._node.create_publisher(Empty, "/sim/release_band", 10)

    def publish_sim_reset(self) -> None:
        """Ask MuJoCo sim to mj_resetData (requires --enable-ros2-sim-reset on sim)."""
        self.pelvis_z = float("inf")
        self._sim_reset_publisher.publish(Empty())

    def publish_release_band(self) -> None:
        """Disable MuJoCo elastic band (requires --enable-ros2-sim-reset on sim)."""
        self._release_band_publisher.publish(Empty())

    # ------------------------------------------------------------------
    # ROS2 callbacks (run on spin thread)
    # ------------------------------------------------------------------

    def _on_body_names(self, msg: String) -> None:
        self.body_names = json.loads(msg.data)
        self._body_names_ready.set()

    def _on_robot_state(self, msg: ByteMultiArray) -> None:
        with self._lock:
            if not self._recording:
                return
            state = msgpack.unpackb(bytes(msg.data), raw=False)
            self._csv_writers["robot_state"].writerow(
                [state["ros_timestamp"], state["index"]]
                + list(state["body_q"])
                + list(state["body_dq"])
                + list(state["last_action"])
                + list(state["base_quat"])
                + list(state["base_ang_vel"])
            )

    def _on_body_poses(self, msg: PoseArray) -> None:
        with self._lock:
            if not self._recording:
                return
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            poses = msg.poses

            if "pelvis" in self.body_names:
                self.pelvis_z = poses[self.body_names.index("pelvis")].position.z

            pos_row = [timestamp]
            quat_row = [timestamp]
            for pose in poses:
                pos_row += [pose.position.x, pose.position.y, pose.position.z]
                quat_row += [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            self._csv_writers["body_pos"].writerow(pos_row)
            self._csv_writers["body_quat"].writerow(quat_row)

    def _on_body_velocities(self, msg: Float64MultiArray) -> None:
        with self._lock:
            if not self._recording:
                return
            timestamp = self._node.get_clock().now().nanoseconds * 1e-9
            data = msg.data
            num_bodies = len(data) // 6
            lin_vel_row = [timestamp]
            ang_vel_row = [timestamp]
            for body_index in range(num_bodies):
                base = body_index * 6
                lin_vel_row += [data[base], data[base + 1], data[base + 2]]
                ang_vel_row += [data[base + 3], data[base + 4], data[base + 5]]
            self._csv_writers["body_lin_vel"].writerow(lin_vel_row)
            self._csv_writers["body_ang_vel"].writerow(ang_vel_row)

    # ------------------------------------------------------------------
    # Trial lifecycle (called from main thread)
    # ------------------------------------------------------------------

    def wait_for_body_names(self, timeout: float = 10.0) -> None:
        self._body_names_ready.wait(timeout=timeout)

    def start_trial(self, trial_dir: str) -> None:
        sim_dir = Path(trial_dir) / "sim"
        robot_dir = Path(trial_dir) / "robot"
        sim_dir.mkdir(parents=True, exist_ok=True)
        robot_dir.mkdir(parents=True, exist_ok=True)

        names = self.body_names

        robot_file = open(robot_dir / "robot_state.csv", "w", newline="")
        robot_writer = csv.writer(robot_file)
        robot_writer.writerow(
            ["ros_timestamp", "index"]
            + [f"q_{i}" for i in range(29)]
            + [f"dq_{i}" for i in range(29)]
            + [f"action_{i}" for i in range(29)]
            + ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"]
            + ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        )

        pos_file = open(sim_dir / "body_pos.csv", "w", newline="")
        pos_writer = csv.writer(pos_file)
        pos_writer.writerow(
            ["ros_timestamp"] + [f"{name}_{axis}" for name in names for axis in ("x", "y", "z")]
        )

        quat_file = open(sim_dir / "body_quat.csv", "w", newline="")
        quat_writer = csv.writer(quat_file)
        quat_writer.writerow(
            ["ros_timestamp"] + [f"{name}_{axis}" for name in names for axis in ("qx", "qy", "qz", "qw")]
        )

        lin_vel_file = open(sim_dir / "body_lin_vel.csv", "w", newline="")
        lin_vel_writer = csv.writer(lin_vel_file)
        lin_vel_writer.writerow(
            ["ros_timestamp"] + [f"{name}_{axis}" for name in names for axis in ("vx", "vy", "vz")]
        )

        ang_vel_file = open(sim_dir / "body_ang_vel.csv", "w", newline="")
        ang_vel_writer = csv.writer(ang_vel_file)
        ang_vel_writer.writerow(
            ["ros_timestamp"] + [f"{name}_{axis}" for name in names for axis in ("wx", "wy", "wz")]
        )

        with self._lock:
            self._csv_files = {
                "robot_state": robot_file,
                "body_pos": pos_file,
                "body_quat": quat_file,
                "body_lin_vel": lin_vel_file,
                "body_ang_vel": ang_vel_file,
            }
            self._csv_writers = {
                "robot_state": robot_writer,
                "body_pos": pos_writer,
                "body_quat": quat_writer,
                "body_lin_vel": lin_vel_writer,
                "body_ang_vel": ang_vel_writer,
            }
            self._recording = True

    def stop_trial(self) -> None:
        with self._lock:
            self._recording = False
            for csv_file in self._csv_files.values():
                csv_file.close()
            self._csv_files.clear()
            self._csv_writers.clear()

    def spin_forever(self) -> None:
        rclpy.spin(self._node)

    def shutdown(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()
