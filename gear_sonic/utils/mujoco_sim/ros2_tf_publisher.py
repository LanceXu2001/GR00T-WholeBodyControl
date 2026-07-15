"""Publish world-to-body TF from MuJoCo pelvis pose (optional ROS 2)."""

from __future__ import annotations

import mujoco


class Ros2TfPublisher:
    """Broadcast one dynamic transform: parent_frame -> child_frame (pelvis body)."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        parent_frame_id: str = "world",
        child_frame_id: str = "pelvis",
        body_name: str = "pelvis",
        node_name: str = "gear_sonic_mujoco_tf",
    ) -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from rclpy.node import Node
        import tf2_ros

        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body {body_name!r} not found in MuJoCo model.")

        self._body_id = body_id
        self._parent_frame_id = parent_frame_id
        self._child_frame_id = child_frame_id

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node(node_name)
        self._broadcaster = tf2_ros.TransformBroadcaster(self._node)
        self._message = TransformStamped()

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish(self, mj_data: mujoco.MjData) -> None:
        position = mj_data.xpos[self._body_id]
        quaternion_wxyz = mj_data.xquat[self._body_id]

        message = self._message
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = self._parent_frame_id
        message.child_frame_id = self._child_frame_id
        message.transform.translation.x = float(position[0])
        message.transform.translation.y = float(position[1])
        message.transform.translation.z = float(position[2])
        message.transform.rotation.x = float(quaternion_wxyz[1])
        message.transform.rotation.y = float(quaternion_wxyz[2])
        message.transform.rotation.z = float(quaternion_wxyz[3])
        message.transform.rotation.w = float(quaternion_wxyz[0])
        self._broadcaster.sendTransform(message)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()
