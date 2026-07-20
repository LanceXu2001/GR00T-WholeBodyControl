/**
 * @file ros2_elevation_cache.hpp
 * @brief Dedicated ROS2 thread that subscribes to a GridMap topic, converts
 *        absolute ground height into policy height-scanner semantics, and
 *        caches the result for the Control thread.
 *
 * Conversion before CNN:
 *   policy_height = sensor_z - ground_z - 0.5
 * where sensor_z is the Z of ``torso_link`` looked up via TF in the height
 * reference frame (default ``world``), and ground_z is the GridMap elevation
 * layer (absolute height in that same Z convention).
 *
 * ## Threading Model
 *
 *   Executor thread: subscribe GridMap → timer parses + TF lookup + convert
 *   Control thread:  try_copy_latest_elevation_row_major() (mutex copy only)
 */

#pragma once

#ifndef ROS2_ELEVATION_CACHE_HPP
#define ROS2_ELEVATION_CACHE_HPP

#if HAS_ROS2

/// Must match the topic published by the elevation map node.
static constexpr const char* kResidualElevationMapTopic = "/elevation_map";

/// Frame whose world/map Z is used as sensor_z (height scanner body).
static constexpr const char* kElevationSensorFrameId = "torso_link";

/// Frame in which absolute ground_z / sensor_z are expressed (matches sim ``world``).
static constexpr const char* kElevationHeightReferenceFrameId = "world";

/// Training offset applied after relative height: sensor_z - ground_z - offset.
static constexpr float kElevationPolicyHeightOffset = 0.5f;

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

/**
 * @class Ros2ElevationCache
 * @brief Subscribes to GridMap, converts to policy height, exposes a snapshot.
 */
class Ros2ElevationCache {
public:
    explicit Ros2ElevationCache(
        const std::string& topic_name,
        int expected_row_count,
        int expected_col_count,
        double cache_update_hz = 50.0,
        const std::string& sensor_frame_id = kElevationSensorFrameId,
        const std::string& height_reference_frame_id = kElevationHeightReferenceFrameId)
        : expected_row_count_(expected_row_count)
        , expected_col_count_(expected_col_count)
        , sensor_frame_id_(sensor_frame_id)
        , height_reference_frame_id_(height_reference_frame_id)
        , has_valid_elevation_(false)
        , is_running_(false)
    {
        if (!rclcpp::ok()) {
            throw std::runtime_error(
                "Ros2ElevationCache requires rclcpp::init() in main() before construction");
        }

        node_ = rclcpp::Node::make_shared("g1_deploy_elevation_ingest");

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        elevation_map_subscriber_ =
            node_->create_subscription<grid_map_msgs::msg::GridMap>(
                topic_name,
                rclcpp::QoS(10),
                [this](grid_map_msgs::msg::GridMap::SharedPtr message) {
                    on_elevation_map_received(std::move(message));
                });
        std::cout << "[Ros2ElevationCache] Subscribed to GridMap topic: "
                  << topic_name << " (QoS depth=10 RELIABLE)" << std::endl;
        std::cout << "[Ros2ElevationCache] Policy height = sensor_z - ground_z - "
                  << kElevationPolicyHeightOffset
                  << " (TF " << height_reference_frame_id_ << " → "
                  << sensor_frame_id_ << ")" << std::endl;

        const auto timer_period_ms = static_cast<int>(1000.0 / cache_update_hz);
        elevation_cache_update_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            [this]() { on_timer_update_cache(); });
        std::cout << "[Ros2ElevationCache] Cache update timer: "
                  << cache_update_hz << " Hz (period " << timer_period_ms << " ms)"
                  << std::endl;

        {
            std::lock_guard<std::mutex> lock(latest_elevation_mutex_);
            latest_elevation_row_major_.assign(
                static_cast<size_t>(expected_row_count_ * expected_col_count_), 0.0f);
        }

        start();
    }

    ~Ros2ElevationCache() {
        stop();
    }

    Ros2ElevationCache(const Ros2ElevationCache&)            = delete;
    Ros2ElevationCache& operator=(const Ros2ElevationCache&) = delete;
    Ros2ElevationCache(Ros2ElevationCache&&)                 = delete;
    Ros2ElevationCache& operator=(Ros2ElevationCache&&)      = delete;

    void start() {
        if (is_running_.load()) {
            return;
        }
        is_running_.store(true);

        executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(node_);

        ros2_spin_thread_ = std::jthread(
            [this](std::stop_token stop_token) {
                spin_loop(stop_token);
            });

        std::cout << "[Ros2ElevationCache] Dedicated spin thread started." << std::endl;
    }

    void stop() {
        if (!is_running_.exchange(false)) {
            return;
        }

        if (executor_) {
            executor_->cancel();
        }

        ros2_spin_thread_ = std::jthread{};

        elevation_cache_update_timer_.reset();
        elevation_map_subscriber_.reset();
        tf_listener_.reset();
        tf_buffer_.reset();

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        node_.reset();
        executor_.reset();

        std::cout << "[Ros2ElevationCache] Dedicated spin thread stopped." << std::endl;
    }

    bool try_copy_latest_elevation_row_major(std::vector<float>& destination) {
        const int expected_element_count = expected_row_count_ * expected_col_count_;
        destination.assign(static_cast<size_t>(expected_element_count), 0.0f);

        if (!has_valid_elevation_.load()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(latest_elevation_mutex_);
        const size_t copy_element_count =
            std::min(latest_elevation_row_major_.size(), destination.size());
        for (size_t i = 0; i < copy_element_count; ++i) {
            destination[i] = latest_elevation_row_major_[i];
        }
        return copy_element_count > 0;
    }

    bool has_valid_elevation() const {
        return has_valid_elevation_.load();
    }

private:
    void on_elevation_map_received(grid_map_msgs::msg::GridMap::SharedPtr message) {
        pending_message_ = std::move(message);
    }

    bool try_lookup_sensor_z(float& sensor_z_out) {
        try {
            const geometry_msgs::msg::TransformStamped transform =
                tf_buffer_->lookupTransform(
                    height_reference_frame_id_,
                    sensor_frame_id_,
                    tf2::TimePointZero);
            sensor_z_out = static_cast<float>(transform.transform.translation.z);
            return std::isfinite(sensor_z_out);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                node_->get_logger(), *node_->get_clock(), 5000,
                "[Ros2ElevationCache] TF %s→%s unavailable (%s); skipping frame.",
                height_reference_frame_id_.c_str(),
                sensor_frame_id_.c_str(),
                ex.what());
            return false;
        }
    }

    void on_timer_update_cache() {
        if (!pending_message_) {
            return;
        }

        grid_map_msgs::msg::GridMap::SharedPtr message_to_process =
            std::move(pending_message_);
        pending_message_ = nullptr;

        size_t elevation_layer_index = message_to_process->layers.size();
        for (size_t i = 0; i < message_to_process->layers.size(); ++i) {
            if (message_to_process->layers[i] == "elevation") {
                elevation_layer_index = i;
                break;
            }
        }
        if (elevation_layer_index >= message_to_process->data.size()) {
            RCLCPP_WARN_THROTTLE(
                node_->get_logger(), *node_->get_clock(), 5000,
                "[Ros2ElevationCache] Received GridMap has no 'elevation' layer; skipping frame.");
            return;
        }

        const std_msgs::msg::Float32MultiArray& elevation_layer =
            message_to_process->data[elevation_layer_index];
        const size_t element_count = elevation_layer.data.size();
        const size_t expected_count =
            static_cast<size_t>(expected_row_count_) * static_cast<size_t>(expected_col_count_);
        if (element_count != expected_count) {
            RCLCPP_WARN_THROTTLE(
                node_->get_logger(), *node_->get_clock(), 5000,
                "[Ros2ElevationCache] Elevation size mismatch: got %zu, expected %zu; skipping.",
                element_count, expected_count);
            return;
        }

        float sensor_z = 0.0f;
        if (!try_lookup_sensor_z(sensor_z)) {
            return;
        }

        // Publisher: grid_map ColMajor flat[ix + iy * n_rows],
        //   ix = forward_from_front, iy = lateral_from_left.
        // CNN: row-major flat[i * n_cols + j], [0,0] = front-left.
        {
            std::lock_guard<std::mutex> lock(latest_elevation_mutex_);
            latest_elevation_row_major_.resize(expected_count);
            const int n_rows = expected_row_count_;
            const int n_cols = expected_col_count_;
            for (int i = 0; i < n_rows; ++i) {
                for (int j = 0; j < n_cols; ++j) {
                    const float ground_z =
                        elevation_layer.data[static_cast<size_t>(i + j * n_rows)];
                    latest_elevation_row_major_[static_cast<size_t>(i * n_cols + j)] =
                        sensor_z - ground_z - kElevationPolicyHeightOffset;
                }
            }
        }

        has_valid_elevation_.store(true);
    }

    void spin_loop(std::stop_token stop_token) {
        while (!stop_token.stop_requested() && is_running_.load() && rclcpp::ok()) {
            executor_->spin_some(std::chrono::milliseconds(10));
        }
    }

    std::shared_ptr<rclcpp::Node>                                    node_;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor>       executor_;
    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr     elevation_map_subscriber_;
    rclcpp::TimerBase::SharedPtr                                     elevation_cache_update_timer_;
    std::shared_ptr<tf2_ros::Buffer>                                 tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                      tf_listener_;

    grid_map_msgs::msg::GridMap::SharedPtr pending_message_;

    std::mutex              latest_elevation_mutex_;
    std::vector<float>      latest_elevation_row_major_;
    std::atomic<bool>       has_valid_elevation_;

    std::jthread        ros2_spin_thread_;
    std::atomic<bool>   is_running_;

    int expected_row_count_;
    int expected_col_count_;
    std::string sensor_frame_id_;
    std::string height_reference_frame_id_;
};

#endif  // HAS_ROS2
#endif  // ROS2_ELEVATION_CACHE_HPP
