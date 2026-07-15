"""Subscribe to /sim/reset and /sim/release_band (optional ROS 2)."""

from __future__ import annotations

from typing import Callable, Optional


class Ros2SimResetSubscriber:
    """Listen for std_msgs/Empty on sim control topics."""

    def __init__(
        self,
        on_reset: Callable[[], None],
        on_release_band: Optional[Callable[[], None]] = None,
        reset_topic: str = "/sim/reset",
        release_band_topic: str = "/sim/release_band",
        node_name: str = "gear_sonic_sim_commands",
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Empty

        self._rclpy_owned_init = not rclpy.ok()
        if self._rclpy_owned_init:
            rclpy.init()
        self._rclpy = rclpy
        self._node: Node = Node(node_name)
        self._node.create_subscription(Empty, reset_topic, lambda _msg: on_reset(), 10)
        if on_release_band is not None:
            self._node.create_subscription(
                Empty, release_band_topic, lambda _msg: on_release_band(), 10
            )

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
            self._node = None  # type: ignore[assignment]
        if self._rclpy_owned_init and self._rclpy.ok():
            self._rclpy.shutdown()
