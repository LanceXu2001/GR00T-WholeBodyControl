cmake_minimum_required(VERSION 3.8)
project(local_elevation_map)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

add_executable(local_elevation_node src/local_elevation_node.cpp)
target_include_directories(local_elevation_node PUBLIC ${PCL_INCLUDE_DIRS})
ament_target_dependencies(local_elevation_node
  rclcpp
  grid_map_msgs
  std_msgs
  tf2_ros
  geometry_msgs
  sensor_msgs
  pcl_conversions
)
target_link_libraries(local_elevation_node ${PCL_LIBRARIES})

install(TARGETS local_elevation_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
