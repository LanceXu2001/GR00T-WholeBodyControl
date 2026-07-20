"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from typing import Dict

import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name
    wbc_config["ENABLE_ROS2_JOINT_STATE"] = config.enable_ros2_joint_state
    wbc_config["ROS2_JOINT_STATE_TOPIC"] = config.ros2_joint_state_topic
    wbc_config["ROS2_JOINT_STATE_RATE_HZ"] = config.ros2_joint_state_rate_hz
    wbc_config["ENABLE_ROS2_TF"] = config.enable_ros2_tf
    wbc_config["ROS2_TF_PARENT_FRAME_ID"] = config.ros2_tf_parent_frame_id
    wbc_config["ROS2_TF_CHILD_FRAME_ID"] = config.ros2_tf_child_frame_id
    wbc_config["ROS2_TF_BODY_NAME"] = config.ros2_tf_body_name
    wbc_config["ROS2_TF_RATE_HZ"] = config.ros2_tf_rate_hz
    wbc_config["ENABLE_ROS2_ELEVATION_MAP"] = config.enable_ros2_elevation_map
    wbc_config["ROS2_ELEVATION_MAP_TOPIC"] = config.ros2_elevation_map_topic
    wbc_config["ROS2_ELEVATION_MAP_BODY_NAME"] = config.ros2_elevation_map_body_name
    wbc_config["ROS2_ELEVATION_MAP_PARENT_FRAME_ID"] = (
        config.ros2_elevation_map_parent_frame_id
    )
    wbc_config["ROS2_ELEVATION_MAP_GRID_FRAME_ID"] = (
        config.ros2_elevation_map_grid_frame_id
    )
    wbc_config["ROS2_ELEVATION_MAP_RATE_HZ"] = config.ros2_elevation_map_rate_hz
    wbc_config["ROS2_ELEVATION_MAP_TERRAIN_GEOM_GROUP"] = (
        config.ros2_elevation_map_terrain_geom_group
    )
    wbc_config["ENABLE_ROS2_LIDAR_POINTCLOUD"] = config.enable_ros2_lidar_pointcloud
    wbc_config["ROS2_LIDAR_POINTCLOUD_TOPIC"] = config.ros2_lidar_pointcloud_topic
    wbc_config["ROS2_LIDAR_POINTCLOUD_RATE_HZ"] = config.ros2_lidar_pointcloud_rate_hz
    wbc_config["LIDAR_SITE_NAME"] = config.lidar_site_name
    wbc_config["LIDAR_SCAN_TYPE"] = config.lidar_scan_type
    wbc_config["LIDAR_BACKEND"] = config.lidar_backend
    wbc_config["ENABLE_ROS2_BODY_STATE"] = config.enable_ros2_body_state
    wbc_config["ROS2_BODY_STATE_TOPIC_PREFIX"] = config.ros2_body_state_topic_prefix
    wbc_config["ROS2_BODY_STATE_RATE_HZ"] = config.ros2_body_state_rate_hz
    wbc_config["ENABLE_ROS2_SIM_RESET"] = config.enable_ros2_sim_reset
    wbc_config["ROS2_SIM_RESET_TOPIC"] = config.ros2_sim_reset_topic
    wbc_config["ROS2_RELEASE_BAND_TOPIC"] = config.ros2_release_band_topic

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
    )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)