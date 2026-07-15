/**
 * @file ros2_elevation_cache.hpp
 * @brief Dedicated ROS2 thread that subscribes to a GridMap topic and caches
 *        the latest elevation layer for lock-free reading by the Control thread.
 *
 * ## Design Summary
 *
 * Problem solved: the previous design tied elevation updates to whichever
 * InputInterface was currently active (e.g. ZMQ, keyboard).  When ZMQ was
 * the active input, the ROS2InputHandler::update() / spin_some() was never
 * called, so the elevation cache went stale.
 *
 * Solution: a dedicated std::jthread runs a SingleThreadedExecutor that owns
 * both the GridMap subscriber and a WallTimer.  This thread is started
 * unconditionally whenever the residual stack is enabled, regardless of the
 * current --input-type selection.
 *
 * ## Threading Model
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ ros2_spin_thread_ (jthread)                                         │
 *   │  SingleThreadedExecutor::spin_some() loop                           │
 *   │  ├─ elevation_map_subscriber_ callback  → store pending_message_   │
 *   │  └─ elevation_cache_update_timer_ callback →                        │
 *   │        read pending_message_, parse elevation,                      │
 *   │        mutex-lock, overwrite latest_elevation_row_major_            │
 *   └─────────────────────────────────────────────────────────────────────┘
 *              ↕ mutex (latest_elevation_mutex_)
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ Control thread (50 Hz)                                              │
 *   │  try_copy_latest_elevation_row_major() → mutex-lock, copy, unlock  │
 *   │  No spin, no executor access.                                       │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * Note: because the subscriber callback and the timer callback both run on
 * the same SingleThreadedExecutor, they are serialized by design.  There is
 * therefore no data race on pending_message_ between those two callbacks; the
 * only shared state that needs a mutex is latest_elevation_row_major_ (shared
 * with the Control thread).
 */

#pragma once

#ifndef ROS2_ELEVATION_CACHE_HPP
#define ROS2_ELEVATION_CACHE_HPP

#if HAS_ROS2

/// ROS2 GridMap topic subscribed by the residual elevation-ingest thread.
/// Must match the topic published by the elevation map node (e.g. gear_sonic's
/// elevation_map_publisher.py).  Written here rather than in the YAML config
/// because it is a deployment constant, not a user-tunable parameter.
static constexpr const char* kResidualElevationMapTopic = "/elevation_map";

#include <atomic>
#include <chrono>
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

/**
 * @class Ros2ElevationCache
 * @brief Subscribes to a ROS2 GridMap topic on a dedicated thread and exposes
 *        a thread-safe snapshot for the Control thread to copy.
 *
 * Lifetime:
 *   1. Construct → creates Node, subscriber, and timer; starts jthread.
 *   2. stop()    → cancels executor, joins thread, resets ROS2 resources.
 *   3. Destruct  → calls stop() if not already called.
 *
 * The Control thread should call try_copy_latest_elevation_row_major() every
 * control tick.  If no valid frame has arrived yet it fills zeros and returns
 * false.
 */
class Ros2ElevationCache {
public:
    /**
     * @brief Construct and start the dedicated elevation-ingest thread.
     *
     * @param topic_name           ROS2 GridMap topic (e.g. "/elevation_map").
     * @param expected_row_count   Number of rows expected from the publisher.
     * @param expected_col_count   Number of columns expected from the publisher.
     * @param cache_update_hz      Rate at which the WallTimer parses the latest
     *                             pending message and writes into the cache.
     *                             Default: 15 Hz (period ≈ 66.7 ms).
     */
    explicit Ros2ElevationCache(
        const std::string& topic_name,
        int expected_row_count,
        int expected_col_count,
        double cache_update_hz = 50.0)
        : expected_row_count_(expected_row_count)
        , expected_col_count_(expected_col_count)
        , has_valid_elevation_(false)
        , is_running_(false)
    {
        if (!rclcpp::ok()) {
            throw std::runtime_error(
                "Ros2ElevationCache requires rclcpp::init() in main() before construction");
        }

        node_ = rclcpp::Node::make_shared("g1_deploy_elevation_ingest");

        // Subscriber: runs on the executor thread; only stores the latest
        // message in pending_message_.  Heavy parsing is deferred to the timer.
        // Match MuJoCo publisher QoS (rclpy create_publisher(..., depth=10) → RELIABLE).
        elevation_map_subscriber_ =
            node_->create_subscription<grid_map_msgs::msg::GridMap>(
                topic_name,
                rclcpp::QoS(10),
                [this](grid_map_msgs::msg::GridMap::SharedPtr message) {
                    on_elevation_map_received(std::move(message));
                });
        std::cout << "[Ros2ElevationCache] Subscribed to GridMap topic: "
                  << topic_name << " (QoS depth=10 RELIABLE)" << std::endl;

        // Timer: parses the latest pending message and writes into the cache.
        const auto timer_period_ms = static_cast<int>(1000.0 / cache_update_hz);
        elevation_cache_update_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            [this]() { on_timer_update_cache(); });
        std::cout << "[Ros2ElevationCache] Cache update timer: "
                  << cache_update_hz << " Hz (period " << timer_period_ms << " ms)" << std::endl;

        // Pre-allocate the cache so the first copy doesn't trigger reallocation.
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

    // Non-copyable, non-movable — owns a thread and ROS2 resources.
    Ros2ElevationCache(const Ros2ElevationCache&)            = delete;
    Ros2ElevationCache& operator=(const Ros2ElevationCache&) = delete;
    Ros2ElevationCache(Ros2ElevationCache&&)                 = delete;
    Ros2ElevationCache& operator=(Ros2ElevationCache&&)      = delete;

    // -----------------------------------------------------------------------
    // Lifecycle API
    // -----------------------------------------------------------------------

    /**
     * @brief Start the dedicated ROS2 executor thread.
     *
     * Called automatically by the constructor; exposed for clarity.
     * Calling start() when already running is a no-op.
     */
    void start() {
        if (is_running_.load()) {
            return;
        }
        is_running_.store(true);

        executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(node_);

        // jthread: requests its own stop token; the spin loop checks it.
        ros2_spin_thread_ = std::jthread(
            [this](std::stop_token stop_token) {
                spin_loop(stop_token);
            });

        std::cout << "[Ros2ElevationCache] Dedicated spin thread started." << std::endl;
    }

    /**
     * @brief Stop the dedicated thread and release all ROS2 resources.
     *
     * Safe to call multiple times; subsequent calls are no-ops.
     * After stop() returns, try_copy_latest_elevation_row_major() will still
     * serve the last cached frame (reads return immediately with the mutex).
     */
    void stop() {
        if (!is_running_.exchange(false)) {
            return;  // Already stopped or never started.
        }

        // Signal the executor to return from spin_some.
        if (executor_) {
            executor_->cancel();
        }

        // Destroying the jthread calls request_stop() then join().
        ros2_spin_thread_ = std::jthread{};

        // Release ROS2 entities in proper DDS teardown order:
        // timer and subscriber before node.
        elevation_cache_update_timer_.reset();
        elevation_map_subscriber_.reset();

        // Give the DDS middleware a moment to clean up internal entities
        // before the node shared_ptr drops to zero.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        node_.reset();
        executor_.reset();

        std::cout << "[Ros2ElevationCache] Dedicated spin thread stopped." << std::endl;
    }

    // -----------------------------------------------------------------------
    // Control-thread read API (never calls spin / executor)
    // -----------------------------------------------------------------------

    /**
     * @brief Copy the latest cached elevation data into @p destination.
     *
     * Called from the Control thread (50 Hz).  Does NOT call spin or touch
     * the ROS2 executor.  The lock is held only for the duration of the copy.
     *
     * If no valid elevation frame has arrived yet, @p destination is filled
     * with zeros and the function returns false.
     *
     * If the cached frame has a different number of elements from
     * expected_row_count * expected_col_count, the available elements are
     * copied and the rest remain zero (same semantics as the previous
     * ROS2InputHandler implementation).
     *
     * @param destination  Output vector; always resized / filled on entry.
     * @return true if at least one valid frame has been cached, false if no
     *         data has arrived yet.
     */
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

    /// True once at least one valid elevation frame has been cached.
    bool has_valid_elevation() const {
        return has_valid_elevation_.load();
    }

private:
    // -----------------------------------------------------------------------
    // Executor thread: subscriber callback
    // Stores the latest incoming message for the timer to process.
    // Both this callback and the timer callback are serialized by the
    // SingleThreadedExecutor, so pending_message_ has no data race between them.
    // -----------------------------------------------------------------------
    void on_elevation_map_received(grid_map_msgs::msg::GridMap::SharedPtr message) {
        pending_message_ = std::move(message);
    }

    // -----------------------------------------------------------------------
    // Executor thread: WallTimer callback (cache_update_hz, default 15 Hz)
    // Parses the latest pending message and writes into the cache.
    // -----------------------------------------------------------------------
    void on_timer_update_cache() {
        if (!pending_message_) {
            // No new message has arrived since the last timer tick — skip.
            return;
        }

        // Take ownership and clear the slot so the next tick starts fresh.
        grid_map_msgs::msg::GridMap::SharedPtr message_to_process =
            std::move(pending_message_);
        pending_message_ = nullptr;

        // Find the "elevation" layer index.
        size_t elevation_layer_index = message_to_process->layers.size();  // sentinel = not found
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

        // Parse the elevation layer and write into the thread-safe cache.
        const std_msgs::msg::Float32MultiArray& elevation_layer =
            message_to_process->data[elevation_layer_index];

        {
            std::lock_guard<std::mutex> lock(latest_elevation_mutex_);
            latest_elevation_row_major_.resize(elevation_layer.data.size());
            for (size_t i = 0; i < elevation_layer.data.size(); ++i) {
                latest_elevation_row_major_[i] = elevation_layer.data[i];
            }
        }

        has_valid_elevation_.store(true);
    }

    // -----------------------------------------------------------------------
    // Dedicated spin loop (runs inside ros2_spin_thread_)
    // -----------------------------------------------------------------------
    void spin_loop(std::stop_token stop_token) {
        while (!stop_token.stop_requested() && is_running_.load() && rclcpp::ok()) {
            executor_->spin_some(std::chrono::milliseconds(10));
        }
    }

    // -----------------------------------------------------------------------
    // ROS2 infrastructure (owned by this object)
    // -----------------------------------------------------------------------
    std::shared_ptr<rclcpp::Node>                                    node_;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor>       executor_;
    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr     elevation_map_subscriber_;
    rclcpp::TimerBase::SharedPtr                                     elevation_cache_update_timer_;

    // -----------------------------------------------------------------------
    // Pending message slot
    // Written by on_elevation_map_received(), read and cleared by
    // on_timer_update_cache().  Both run on the same executor thread → no
    // mutex required between them.
    // -----------------------------------------------------------------------
    grid_map_msgs::msg::GridMap::SharedPtr pending_message_;

    // -----------------------------------------------------------------------
    // Latest parsed elevation cache
    // Written by on_timer_update_cache() (executor thread).
    // Read by try_copy_latest_elevation_row_major() (Control thread).
    // Protected by latest_elevation_mutex_.
    // -----------------------------------------------------------------------
    std::mutex              latest_elevation_mutex_;
    std::vector<float>      latest_elevation_row_major_;
    std::atomic<bool>       has_valid_elevation_;

    // -----------------------------------------------------------------------
    // Thread and lifecycle state
    // -----------------------------------------------------------------------
    std::jthread        ros2_spin_thread_;
    std::atomic<bool>   is_running_;

    int expected_row_count_;
    int expected_col_count_;
};

#endif  // HAS_ROS2
#endif  // ROS2_ELEVATION_CACHE_HPP
