Terminal 1:
launch lidar driver in ros2_ws
```
ros2 launch livox_ros_driver2 rviz_MID360_launch.py 
```

Terminal 2:
launch tf publisher in ros2_ws
```
ros2 launch lidar_localization_ros2 mid360_legged_localization.launch.py map_path:=/home/malab/XJL/project/GR00T-WholeBodyControl/gear_sonic_deploy/map/Intentele/Preview.pcd cloud_topic:=/livox/lidar imu_topic:=/livox/imu enable_map_odom_tf:=false set_initial_pose:=false
```

Terminal 3:
launch elevation publisher in ros2_ws
```
ros2 launch local_elevation_map local_elevation.launch.py
```

Terminal 4:
start controller
```
bash deploy.sh real
```