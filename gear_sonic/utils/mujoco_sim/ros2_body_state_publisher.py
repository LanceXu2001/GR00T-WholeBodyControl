"""Publish per-body world-frame pose and velocity from MuJoCo (optional ROS 2).

Topics published:
    {prefix}/poses         geometry_msgs/PoseArray      position + orientation for each body
    {prefix}/velocities    std_msgs/Float64MultiArray   [lin_vel(3), ang_vel(3)] per body, flat
    {prefix}/names         std_msgs/String              JSON list of body names (latched once)

Body ordering follows MuJoCo body index; the worldbody (index 0) is excluded.

Velocity convention (world frame, at body CoM):
    data[i*6 + 0:3]  = mj_data.cvel[body_id, 3:6]  — CoM linear  velocity (m/s)
    data[i*6 + 3:6]  = mj_data.cvel[body_id, 0:3]  — angular velocity (rad/s)

ROS quaternion uses (x, y, z, w); MuJoCo stores (w, x, y, z) — this class converts.
"""

from __future__ import annotations

import json
from typing import List, Optional

import mujoco


class Ros2BodyStatePublisher:
    """Broadcast world-frame pose and velocity for every non-world MuJoCo body."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        topic_prefix: str = "/sim/bodies",
        node_name: str = "gear_sonic_body_state",
        qos_depth: int = 10,
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Pose, PoseArray
            from rclpy.node import Node
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile
            from std_msgs.msg import Float64MultiArray, String
        except ImportError as exc:
            raise ImportError(
                "ROS 2 Python packages not found (rclpy, geometry_msgs, std_msgs). "
                "Source your ROS 2 workspace overlay, then re-run with "
                "--enable-ros2-body-state."
            ) from exc

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()
        self._rclpy = rclpy
        self._node: Node = Node(node_name)

        # Collect all non-world body indices (worldbody = 0 has no meaningful velocity)
        self._body_ids: List[int] = list(range(1, mj_model.nbody))
        self._body_names: List[str] = [
            mj_model.body(b).name or f"body_{b}" for b in self._body_ids
        ]
        n = len(self._body_ids)

        # Pose publisher: geometry_msgs/PoseArray (pos + quat per body)
        self._pub_poses = self._node.create_publisher(
            PoseArray, f"{topic_prefix}/poses", qos_depth
        )

        # Velocity publisher: flat Float64MultiArray, n*6 elements
        self._pub_vel = self._node.create_publisher(
            Float64MultiArray, f"{topic_prefix}/velocities", qos_depth
        )

        # Names publisher: latched String (JSON list), published once at startup
        names_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        pub_names = self._node.create_publisher(
            String, f"{topic_prefix}/names", names_qos
        )
        names_msg = String()
        names_msg.data = json.dumps(self._body_names)
        pub_names.publish(names_msg)

        # Pre-allocate reusable message objects to avoid per-tick allocations
        self._pose_msg = PoseArray()
        self._pose_msg.poses = [Pose() for _ in range(n)]
        self._vel_msg = Float64MultiArray()
        self._vel_msg.data = [0.0] * (n * 6)
        self._n = n

    # ------------------------------------------------------------------

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish(self, mj_data: mujoco.MjData) -> None:
        stamp = self._node.get_clock().now().to_msg()
        self._pose_msg.header.stamp = stamp
        self._pose_msg.header.frame_id = "world"

        vel_data = self._vel_msg.data
        poses = self._pose_msg.poses

        for i, body_id in enumerate(self._body_ids):
            pos = mj_data.xpos[body_id]        # world-frame CoM position
            q_wxyz = mj_data.xquat[body_id]    # world-frame orientation (w,x,y,z)
            cvel = mj_data.cvel[body_id]        # [ang(3), lin(3)] world frame, at CoM

            p = poses[i]
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = float(pos[2])
            # ROS quaternion convention: (x, y, z, w)
            p.orientation.x = float(q_wxyz[1])
            p.orientation.y = float(q_wxyz[2])
            p.orientation.z = float(q_wxyz[3])
            p.orientation.w = float(q_wxyz[0])

            base = i * 6
            # lin_vel (world frame, m/s)
            vel_data[base + 0] = float(cvel[3])
            vel_data[base + 1] = float(cvel[4])
            vel_data[base + 2] = float(cvel[5])
            # ang_vel (world frame, rad/s)
            vel_data[base + 3] = float(cvel[0])
            vel_data[base + 4] = float(cvel[1])
            vel_data[base + 5] = float(cvel[2])

        self._pub_poses.publish(self._pose_msg)
        self._pub_vel.publish(self._vel_msg)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()
