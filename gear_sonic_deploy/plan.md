# 墙钟数据采集 Pipeline 计划（ROS2 版）

`reset → 发布参考轨迹 → ROS2 采集 → 判定结束 → 下一 trial`

前提：**`HAS_ROS2=1` 重编 deploy**（开 elevation map 本来就要改），之后 sim、deploy、编排脚本三者共享同一个 ROS2 DDS 域，数据全部走话题，不再需要 CSV 拷贝。

---

## 1. 数据流总览

```text
run_sim_loop.py  (MuJoCo 200 Hz, 墙钟)
    PUB /tf                    world → pelvis   (50 Hz，fall 判定用)
    PUB /joint_states          关节角 / 速度     (可选，与 deploy body_q 对比)
    PUB /elevation_map         GridMap          → deploy Ros2ElevationCache (residual 输入)
    PUB /sim/bodies/poses      geometry_msgs/PoseArray  所有 body 世界系位置+姿态 (≤200 Hz)
    PUB /sim/bodies/velocities std_msgs/Float64MultiArray  [lin_vel(3),ang_vel(3)] per body
    PUB /sim/bodies/names      std_msgs/String  JSON body 名列表，latched 一次

deploy  (50 Hz, HAS_ROS2=1)
    SUB /elevation_map                   → residual 高程输入
    PUB G1Env/env_state_act  ByteMultiArray (msgpack, 50 Hz)
        body_q[29], body_dq[29], last_action[29]
        base_quat[4], base_ang_vel[3]
        token_state[N], index, ros_timestamp

batch_motion_collect.py  (编排脚本)
    ZMQ PUB :5556  →  参考 pose（路径 1，脚本内墙钟发包）
    ZMQ PUB        →  command start/stop
    SUB G1Env/env_state_act      →  robot_*.csv（50 Hz，deploy 控制器视角）
    SUB /tf                      →  fall 判定（pelvis_z，50 Hz）
    SUB /sim/bodies/poses        →  sim/body_pos.csv  所有 body 世界系位置+姿态
    SUB /sim/bodies/velocities   →  sim/body_vel.csv  所有 body 线速度+角速度（物理值）
    SUB /joint_states            →  sim/joint_states.csv（可选）
```

---

## 2. 文件规划（4 个文件）

| 路径 | 职责 |
|------|------|
| `gear_sonic_deploy/scripts/batch_motion_collect.py` | CLI 入口、trial 状态机、进度、`manifest.json` |
| `gear_sonic_deploy/scripts/motion_collect/config.py` | 配置 dataclass + YAML 加载 |
| `gear_sonic_deploy/scripts/motion_collect/reference_publisher.py` | npz 加载、插值、墙钟 ZMQ PUB + 直接写 `ref_*.csv` |
| `gear_sonic_deploy/scripts/motion_collect/ros2_collector.py` | rclpy 节点：SUB `G1Env/env_state_act` + `/tf` + `/sim/bodies/{poses,velocities}`，按 trial 写 CSV |
| `gear_sonic_deploy/config/motion_collect_example.yaml` | 示例配置 |

---

## 3. 配置示例（YAML）

```yaml
output:
  root_dir: "collect/my_experiment_001"   # 用户自定义，支持 {date} 占位

reference:
  npz_path: "gear_sonic_deploy/reference/motions/walk_forward.npz"
  target_fps: 50.0                         # 与 deploy 控制频率一致

collection:
  num_trials: 100
  trial_timeout_sec: 120.0

initial_pose:
  pelvis_xyz: [0.0, 0.0, 0.75]            # 每 trial 固定初始位姿
  pelvis_yaw_rad: 0.0

record_fields:
  # 参考（reference_publisher 直接写，不走 ROS2）
  - ref_frame_index
  - ref_smpl_joints
  - ref_joint_pos
  # 机器人（SUB G1Env/env_state_act，50 Hz，deploy 控制器视角）
  - robot_joint_pos    # body_q[29]
  - robot_joint_vel    # body_dq[29]
  - robot_action       # last_action[29]
  - robot_base_quat    # base_quat[4]
  - robot_base_ang_vel # base_ang_vel[3]
  - robot_token        # token_state[N]
  # 仿真全 body（SUB /sim/bodies/poses + /sim/bodies/velocities，≤200 Hz，物理真值）
  - sim_body_pos       # 所有 body 世界系位置 (nbody-1, 3)
  - sim_body_quat      # 所有 body 世界系四元数 (nbody-1, 4) xyzw
  - sim_body_lin_vel   # 所有 body CoM 线速度 (nbody-1, 3)，来自 mj_data.cvel
  - sim_body_ang_vel   # 所有 body 角速度 (nbody-1, 3)，来自 mj_data.cvel
  # fall 判定仅用 /tf，不单独记录（pelvis_z 可从 sim_body_pos 中的 pelvis 行取得）

stop_conditions:
  fall_height_threshold: 0.2   # pelvis z < 此值 → fallen（与 base_sim.check_fall 一致）
  ref_idle_sec: 2.0            # frame_index 停止递增超过此秒 → reference_finished
  trial_timeout_sec: 120.0

processes:
  start_sim: true
  start_deploy: true
  sim_args:
    - "--enable-ros2-tf"
    - "--enable-ros2-elevation-map"
    - "--enable-ros2-body-state"
    - "--ros2-body-state-rate-hz=200"   # 物理步全频，采集脚本按需下采样
  deploy_args: []              # HAS_ROS2=1 编译后无需额外参数

progress:
  write_manifest: true
```

---

## 4. Trial 状态机

```text
FOR trial_id IN 0 .. num_trials-1:

  PRINT "trial {trial_id+1}/{num_trials}"

  1. RESET
     - ZMQ command: stop → start（deploy motion 游标归零）
     - ros2_collector: 清空本 trial 缓冲，创建 trial_{id:03d}/ 目录
     - reference_publisher: frame_index 归零

  2. PLAY（同时开始）
     - reference_publisher 线程：墙钟 50 Hz 发 ZMQ pose + 写 ref_*.csv
     - ros2_collector 线程：
         SUB G1Env/env_state_act        → robot_*.csv（50 Hz）
         SUB /tf                        → 监测 pelvis_z（fall 判定，50 Hz）
         SUB /sim/bodies/poses          → sim/body_pos.csv（≤200 Hz）
         SUB /sim/bodies/velocities     → sim/body_vel.csv（≤200 Hz，物理速度）

  3. WAIT（直到任一条件满足）
     - pelvis_z < fall_height_threshold       → end_reason = fallen
     - frame_index 停止递增 ≥ ref_idle_sec   → end_reason = reference_finished
     - elapsed ≥ trial_timeout_sec            → end_reason = timeout
     - Ctrl-C                                 → end_reason = aborted

  4. STOP
     - 停止 reference_publisher 发包（ZMQ command: stop）
     - ros2_collector flush CSV
     - 写 trial_meta.json: {trial_id, end_reason, wall_duration_sec,
                             ref_frames, fall_frame, initial_pose}

  5. PROGRESS
     - 更新 manifest.json
     - 打印 "37/100  ok/fallen/timeout"
```

---

## 5. 输出目录结构

```text
{root_dir}/
  manifest.json
  config_resolved.yaml
  trial_000/
    trial_meta.json
    reference/
      ref_frame_index.csv
      ref_smpl_joints.csv      # frame_index, j0_x, j0_y, j0_z, ...
    robot/
      body_q.csv               # index, ros_timestamp, q0..q28
      body_dq.csv
      last_action.csv
      base_quat.csv
      base_ang_vel.csv
      token_state.csv
    sim/
      body_pos.csv             # ros_timestamp, {body_name}_x, {body_name}_y, {body_name}_z, ...
      body_quat.csv            # ros_timestamp, {body_name}_qx, _qy, _qz, _qw, ...
      body_lin_vel.csv         # ros_timestamp, {body_name}_vx, _vy, _vz, ...
      body_ang_vel.csv         # ros_timestamp, {body_name}_wx, _wy, _wz, ...
      # 列名由 /sim/bodies/names（latched）在启动时一次性读取后生成
  trial_001/
    ...
```

`manifest.json` 关键字段：
```json
{
  "total_trials": 100,
  "completed_trials": 37,
  "trials": [
    {"id": 0, "end_reason": "reference_finished", "wall_sec": 8.2},
    {"id": 1, "end_reason": "fallen",             "wall_sec": 3.1}
  ]
}
```

---

## 6. 前置条件与启动

### 编译（一次性）
```bash
export HAS_ROS2=1
cd gear_sonic_deploy && cmake -B build -DHAS_ROS2=1 && cmake --build build -j
```

### 启动顺序（编排脚本自动 Popen，或手动三终端）
```bash
# 终端 1：仿真
python gear_sonic/scripts/run_sim_loop.py \
  --enable-ros2-tf \
  --enable-ros2-elevation-map \
  --enable-ros2-body-state \
  --ros2-body-state-rate-hz 200

# 终端 2：控制器（HAS_ROS2=1 编译版）
bash gear_sonic_deploy/deploy.sh sim --input-type zmq

# 终端 3：采集
python gear_sonic_deploy/scripts/batch_motion_collect.py \
  --config gear_sonic_deploy/config/motion_collect_example.yaml
```

---

## 7. 实现阶段

### Phase 0 — 手工验证（无新脚本）
- [ ] 确认 `HAS_ROS2=1` 编译通过，`/elevation_map` → deploy residual 有效
- [ ] `ros2 topic echo G1Env/env_state_act` 能收到 msgpack 包
- [ ] `ros2 topic echo /tf` 能看到 `world → pelvis`
- [ ] `ros2 topic echo /sim/bodies/poses` 能看到所有 body 位置（`--enable-ros2-body-state`）
- [ ] `ros2 topic hz /sim/bodies/velocities` 确认频率达到配置值
- [ ] 手工跑一次 `amass_zmq_player`，记录单次 trial 墙钟时长

### Phase 1 — MVP（约 400 行 Python）
- [ ] `config.py` + `motion_collect_example.yaml`
- [ ] `reference_publisher.py`：npz 插值 + 墙钟 ZMQ PUB + 写 `ref_*.csv`
- [ ] `ros2_collector.py`：
  - 启动时 SUB `/sim/bodies/names`（latched）→ 解析 body 名列表，生成 CSV 列头
  - SUB `G1Env/env_state_act`（msgpack decode）→ `robot_*.csv`（50 Hz）
  - SUB `/tf` → fall 判定（pelvis_z）
  - SUB `/sim/bodies/poses` + `/sim/bodies/velocities` → `sim/body_{pos,quat,lin_vel,ang_vel}.csv`
- [ ] `batch_motion_collect.py`：trial 循环、进度、`manifest.json`

### Phase 2 — 完善
- [ ] SUB `/joint_states`（可选对比字段）
- [ ] `initial_pose` 每 trial 注入（sim API 或配置）
- [ ] `random_xy_yaw` 初始位姿分布
- [ ] 断点续采（`start_trial_index` 配置）

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| `HAS_ROS2=1` 重编引入新 bug | Phase 0 手工验证链路；先单独测 `G1Env/env_state_act` 话题 |
| msgpack decode 字段对不上 | `ros2_collector` 启动时打印一条 decode 样例，Phase 0 确认字段名 |
| ROS2 时间戳与 ZMQ frame_index 对齐 | 每条记录同时保存 `ros_timestamp`（deploy）和 `frame_index`（参考），后处理按时间戳插值 |
| `/sim/bodies/*` 与 deploy 频率不同（200 vs 50 Hz）| 两路数据各自打时间戳独立写 CSV，后处理按 `ros_timestamp` 插值对齐；fall 判定仍走 50 Hz 的 `/tf` |
| 100 次墙钟耗时长 | `manifest.json` 支持断点续采 |
