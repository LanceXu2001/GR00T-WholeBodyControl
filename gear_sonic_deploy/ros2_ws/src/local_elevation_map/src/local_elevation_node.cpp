#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>
#include <std_msgs/msg/multi_array_layout.hpp>

/**
 * Local elevation map node aligned with MuJoCo ElevationMapPublisher semantics:
 *
 * - Heading-aligned local frame: +X_local = forward (yaw only), +Y_local = left.
 * - Flat buffer is F-order of height[forward, lateral]:
 *     k = forward + lateral * n_forward
 *   so cell 0 is the rear-right corner (-half_f, -half_l).
 * - Elevation values are relative to the robot sensor (MuJoCo torso / robot frame):
 *     sensor_z = base_link_z - lidar_height_above_robot
 *     elevation = sensor_z - ground_z - height_offset
 *   On the real robot, base_link is the lidar pose; lidar sits
 *   lidar_height_above_robot above the robot frame (default 0.47618 m).
 *   MuJoCo uses height_offset = 0.5. Missing cells publish 0.0.
 * - Publishes grid_map_msgs/GridMap on /elevation_map with layer "elevation".
 */
class LocalElevationNode : public rclcpp::Node
{
public:
  LocalElevationNode()
  : Node("local_elevation_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    pcd_path_ = declare_parameter<std::string>("pcd_path", "");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    length_x_ = declare_parameter<double>("length_x", 1.5);   // forward extent
    length_y_ = declare_parameter<double>("length_y", 1.5);   // lateral extent
    resolution_ = declare_parameter<double>("resolution", 0.1);
    publish_rate_ = declare_parameter<double>("publish_rate", 10.0);
    use_min_z_ = declare_parameter<bool>("use_min_z", true);
    min_points_per_cell_ = declare_parameter<int>("min_points_per_cell", 1);
    z_min_ = declare_parameter<double>("z_min", -2.0);
    z_max_ = declare_parameter<double>("z_max", 2.0);
    // Match MuJoCo: sensor_z - ground_z - 0.5
    height_offset_ = declare_parameter<double>("height_offset", 0.5);
    // base_link is lidar pose; lidar is this far above the robot/MuJoCo sensor frame.
    lidar_height_above_robot_ =
      declare_parameter<double>("lidar_height_above_robot", 0.47618);
    output_topic_ = declare_parameter<std::string>("output_topic", "/elevation_map");
    layer_name_ = declare_parameter<std::string>("layer_name", "elevation");

    if (pcd_path_.empty()) {
      throw std::runtime_error("Parameter 'pcd_path' must be set");
    }
    if (resolution_ <= 0.0 || length_x_ <= 0.0 || length_y_ <= 0.0) {
      throw std::runtime_error("length_x/length_y/resolution must be > 0");
    }

    // Same cell counting convention as MuJoCo: length = n_cells * resolution,
    // sample centers span ±(n-1)/2 * resolution.
    n_forward_ = static_cast<int>(std::lround(length_x_ / resolution_));
    n_lateral_ = static_cast<int>(std::lround(length_y_ / resolution_));
    if (n_forward_ < 1 || n_lateral_ < 1) {
      throw std::runtime_error("Derived grid size must be >= 1");
    }
    forward_half_span_ = 0.5 * static_cast<double>(n_forward_ - 1) * resolution_;
    lateral_half_span_ = 0.5 * static_cast<double>(n_lateral_ - 1) * resolution_;
    total_cells_ = n_forward_ * n_lateral_;

    loadGlobalCloud();
    buildMessageTemplate();

    // Match MuJoCo / deploy Ros2ElevationCache: RELIABLE depth=10
    map_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(output_topic_, rclcpp::QoS(10));

    const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_, 0.1));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&LocalElevationNode::onTimer, this));

    RCLCPP_INFO(
      get_logger(),
      "Local elevation ready: grid=%dx%d (F×L) res=%.3fm "
      "rear-right=first, sensor_z=base_z-%.5f, height_offset=%.3f, topic=%s",
      n_forward_, n_lateral_, resolution_, lidar_height_above_robot_,
      height_offset_, output_topic_.c_str());
  }

private:
  void loadGlobalCloud()
  {
    RCLCPP_INFO(get_logger(), "Loading PCD: %s", pcd_path_.c_str());
    pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud;
    if (pcl::io::loadPCDFile(pcd_path_, rgb_cloud) < 0) {
      pcl::PointCloud<pcl::PointXYZ> xyz_cloud;
      if (pcl::io::loadPCDFile(pcd_path_, xyz_cloud) < 0) {
        throw std::runtime_error("Failed to load PCD: " + pcd_path_);
      }
      cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
      *cloud_ = xyz_cloud;
    } else {
      cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
      cloud_->reserve(rgb_cloud.size());
      for (const auto & p : rgb_cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
          continue;
        }
        cloud_->push_back(pcl::PointXYZ(p.x, p.y, p.z));
      }
      cloud_->width = static_cast<uint32_t>(cloud_->size());
      cloud_->height = 1;
      cloud_->is_dense = false;
    }

    RCLCPP_INFO(get_logger(), "Loaded %zu valid points", cloud_->size());
  }

  void buildMessageTemplate()
  {
    // Same layout as gear_sonic/utils/mujoco_sim/elevation_map_publisher.py
    std_msgs::msg::MultiArrayLayout elevation_layout;
    std_msgs::msg::MultiArrayDimension dim_col;
    dim_col.label = "column_index";
    dim_col.size = static_cast<uint32_t>(n_lateral_);
    dim_col.stride = static_cast<uint32_t>(total_cells_);
    std_msgs::msg::MultiArrayDimension dim_row;
    dim_row.label = "row_index";
    dim_row.size = static_cast<uint32_t>(n_forward_);
    dim_row.stride = static_cast<uint32_t>(n_forward_);
    elevation_layout.dim = {dim_col, dim_row};
    elevation_layout.data_offset = 0;

    elevation_data_.assign(static_cast<size_t>(total_cells_), 0.0f);
    std_msgs::msg::Float32MultiArray elevation_layer;
    elevation_layer.layout = elevation_layout;
    elevation_layer.data = elevation_data_;

    grid_map_msg_.header.frame_id = base_frame_;
    grid_map_msg_.info.resolution = static_cast<float>(resolution_);
    grid_map_msg_.info.length_x = static_cast<float>(n_forward_ * resolution_);
    grid_map_msg_.info.length_y = static_cast<float>(n_lateral_ * resolution_);
    grid_map_msg_.info.pose.orientation.w = 1.0;
    grid_map_msg_.layers = {layer_name_};
    grid_map_msg_.basic_layers = {layer_name_};
    grid_map_msg_.outer_start_index = 0;
    grid_map_msg_.inner_start_index = 0;
    grid_map_msg_.data = {elevation_layer};
  }

  static double yawFromQuat(double x, double y, double z, double w)
  {
    // geometry_msgs uses (x, y, z, w); match MuJoCo yaw-from-quat about +Z.
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }

  void onTimer()
  {
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for TF %s -> %s: %s", map_frame_.c_str(), base_frame_.c_str(), ex.what());
      return;
    }

    const double robot_x = tf.transform.translation.x;
    const double robot_y = tf.transform.translation.y;
    // base_link_z is the lidar; MuJoCo sensor_z is the robot frame below it.
    const double base_link_z = tf.transform.translation.z;
    const double sensor_z = base_link_z - lidar_height_above_robot_;
    const double yaw = yawFromQuat(
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z,
      tf.transform.rotation.w);
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    // Coarse axis-aligned crop in map frame (padding for yaw-aligned window).
    const double half_diag =
      0.5 * std::hypot(n_forward_ * resolution_, n_lateral_ * resolution_) + resolution_;
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(cloud_);
    crop.setMin(
      Eigen::Vector4f(
        static_cast<float>(robot_x - half_diag),
        static_cast<float>(robot_y - half_diag),
        static_cast<float>(sensor_z + z_min_), 1.0f));
    crop.setMax(
      Eigen::Vector4f(
        static_cast<float>(robot_x + half_diag),
        static_cast<float>(robot_y + half_diag),
        static_cast<float>(sensor_z + z_max_), 1.0f));

    pcl::PointCloud<pcl::PointXYZ> local_cloud;
    crop.filter(local_cloud);

    std::vector<float> accum(static_cast<size_t>(total_cells_), 0.0f);
    std::vector<int> counts(static_cast<size_t>(total_cells_), 0);
    std::vector<float> min_z(
      static_cast<size_t>(total_cells_), std::numeric_limits<float>::infinity());

    for (const auto & pt : local_cloud.points) {
      const double dx = pt.x - robot_x;
      const double dy = pt.y - robot_y;
      // World → heading-aligned local (yaw only): +X forward, +Y left.
      const double local_forward = cos_yaw * dx + sin_yaw * dy;
      const double local_lateral = -sin_yaw * dx + cos_yaw * dy;

      const int i_f = static_cast<int>(std::lround(
        (local_forward + forward_half_span_) / resolution_));
      const int i_l = static_cast<int>(std::lround(
        (local_lateral + lateral_half_span_) / resolution_));
      if (i_f < 0 || i_f >= n_forward_ || i_l < 0 || i_l >= n_lateral_) {
        continue;
      }

      // F-order: forward varies fastest → rear-right is index 0.
      const size_t linear = static_cast<size_t>(i_f + i_l * n_forward_);
      counts[linear] += 1;
      accum[linear] += pt.z;
      min_z[linear] = std::min(min_z[linear], pt.z);
    }

    // Relative-to-sensor elevation; empty cells stay 0.0 (MuJoCo no-hit).
    elevation_data_.assign(static_cast<size_t>(total_cells_), 0.0f);
    for (int linear = 0; linear < total_cells_; ++linear) {
      const size_t idx = static_cast<size_t>(linear);
      if (counts[idx] < min_points_per_cell_) {
        continue;
      }
      const float ground_z = use_min_z_ ?
        min_z[idx] :
        (accum[idx] / static_cast<float>(counts[idx]));
      elevation_data_[idx] =
        static_cast<float>(sensor_z - static_cast<double>(ground_z) - height_offset_);
    }

    grid_map_msg_.header.stamp = now();
    grid_map_msg_.header.frame_id = base_frame_;
    grid_map_msg_.info.pose.position.x = 0.0;
    grid_map_msg_.info.pose.position.y = 0.0;
    grid_map_msg_.info.pose.position.z = 0.0;
    grid_map_msg_.info.pose.orientation.x = 0.0;
    grid_map_msg_.info.pose.orientation.y = 0.0;
    grid_map_msg_.info.pose.orientation.z = 0.0;
    grid_map_msg_.info.pose.orientation.w = 1.0;
    grid_map_msg_.data[0].data = elevation_data_;
    map_pub_->publish(grid_map_msg_);
  }

  std::string pcd_path_;
  std::string map_frame_;
  std::string base_frame_;
  double length_x_{1.5};
  double length_y_{1.5};
  double resolution_{0.1};
  double publish_rate_{10.0};
  bool use_min_z_{true};
  int min_points_per_cell_{1};
  double z_min_{-2.0};
  double z_max_{2.0};
  double height_offset_{0.5};
  double lidar_height_above_robot_{0.47618};
  std::string output_topic_;
  std::string layer_name_;

  int n_forward_{15};
  int n_lateral_{15};
  int total_cells_{225};
  double forward_half_span_{0.7};
  double lateral_half_span_{0.7};

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  grid_map_msgs::msg::GridMap grid_map_msg_;
  std::vector<float> elevation_data_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<LocalElevationNode>());
  } catch (const std::exception & e) {
    fprintf(stderr, "local_elevation_node failed: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
