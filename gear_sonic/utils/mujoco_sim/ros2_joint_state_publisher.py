"""Publish sensor_msgs/JointState from MuJoCo hinge and slide joints (optional ROS 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mujoco


@dataclass(frozen=True)
class _JointSlot:
    name: str
    qpos_adr: int
    dof_adr: int


def _hinge_and_slide_slots(mj_model: mujoco.MjModel) -> List[_JointSlot]:
    """Collect scalar joints only (skip free / ball roots)."""
    FREE = mujoco.mjtJoint.mjJNT_FREE
    BALL = mujoco.mjtJoint.mjJNT_BALL
    out: List[_JointSlot] = []
    for j in range(mj_model.njnt):
        jt = mj_model.jnt_type[j]
        if jt == FREE or jt == BALL:
            continue
        name = mj_model.joint(j).name
        out.append(
            _JointSlot(
                name=name if name else f"joint_{j}",
                qpos_adr=int(mj_model.jnt_qposadr[j]),
                dof_adr=int(mj_model.jnt_dofadr[j]),
            )
        )
    return out


class Ros2JointStatePublisher:
    """Thin wrapper: one publisher, fill JointState from mjData each step."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        topic: str,
        node_name: str = "gear_sonic_mujoco_joint_states",
        qos_depth: int = 10,
    ) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as e:  # pragma: no cover - requires ROS 2 Python
            raise ImportError(
                "ROS 2 Python packages not found (rclpy, sensor_msgs). "
                "Source your ROS 2 workspace overlay, then re-run with "
                "--enable-ros2-joint-state."
            ) from e

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()
        self._rclpy = rclpy
        self._node: Node = Node(node_name)
        self._pub = self._node.create_publisher(JointState, topic, qos_depth)
        self._msg = JointState()
        self._slots = _hinge_and_slide_slots(mj_model)
        self._msg.name = [s.name for s in self._slots]

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish(self, mj_data: mujoco.MjData) -> None:
        stamp = self._node.get_clock().now().to_msg()
        self._msg.header.stamp = stamp
        self._msg.header.frame_id = ""
        self._msg.position = [float(mj_data.qpos[s.qpos_adr]) for s in self._slots]
        self._msg.velocity = [float(mj_data.qvel[s.dof_adr]) for s in self._slots]
        self._msg.effort = []
        self._pub.publish(self._msg)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()
