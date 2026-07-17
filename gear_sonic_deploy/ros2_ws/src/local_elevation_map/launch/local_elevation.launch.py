import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('local_elevation_map')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')
    default_viz = os.path.join(pkg_share, 'config', 'visualization.yaml')

    params_file = LaunchConfiguration('params_file')
    viz_config = LaunchConfiguration('viz_config')
    use_viz = LaunchConfiguration('use_viz')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('viz_config', default_value=default_viz),
        DeclareLaunchArgument('use_viz', default_value='true'),

        Node(
            package='local_elevation_map',
            executable='local_elevation_node',
            name='local_elevation_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='grid_map_visualization',
            executable='grid_map_visualization',
            name='grid_map_visualization',
            output='screen',
            parameters=[viz_config],
            condition=IfCondition(use_viz),
        ),
    ])
