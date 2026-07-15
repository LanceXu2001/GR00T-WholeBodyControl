#!/usr/bin/env python3
"""AMASS motion ZMQ player.

Publishes motion clips listed in intentele ``test_dataset.yaml``
as a Pico-compatible ZMQ stream (protocol version 3, PUB socket port 5556).

Required npz keys (same three keys as intentele ``MotionLibNPZ`` / ``prepare_dataset`` style):
  fps               scalar > 0          clip frame rate (Hz)
  joint_poses       float32 [T, 24, 3]  world-space joint positions
  root_orientation  float32 [T, 4]      root quaternion on disk: use ``--root-orientation-disk-order``

``smpl_joints`` and the root row of ``body_quat_w`` follow the same construction as
``intentele/.../mdp/commands.py`` → ``MotionCommand._apply_heading_alignment`` with
``heading_delta_quaternion = identity`` (no live robot, so no heading alignment to the robot reset pose).

Key bindings:
  T / t    play / resume (start streaming; clears hold-first-frame)
  R / r    restart current clip from frame 0
  N / n    next clip
  P / p    previous clip
  Ctrl-C   quit

Requires deploy with ``--input-type zmq_manager``.  T/R switch to streamed motion;
each clip end switches back to planner (same as pressing Enter twice on controller).

Usage (from repo root):
  python gear_sonic_deploy/amass_zmq_player.py
  python gear_sonic_deploy/amass_zmq_player.py --target-fps 50   # uniform output timeline (default 50)
  python gear_sonic_deploy/amass_zmq_player.py --vis-smpl   # optional SMPL-24 skeleton window
  python gear_sonic_deploy/amass_zmq_player.py --root-orientation-disk-order wxyz   # if your npz stores scalar-first

Default ``--root-orientation-disk-order xyzw`` matches intentele ``motion_lib_smpl_npz.py`` (scalar last on disk).
"""

from __future__ import annotations

import argparse
import atexit
import pathlib
import queue
import sys
import termios
import threading
import time
import tty
from typing import List, Optional, Tuple

# Allow running from gear_sonic_deploy/ without pip install -e gear_sonic/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

try:
    import zmq
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"No module named 'zmq' (pyzmq) for interpreter {sys.executable}. "
        "Activate the sonic env and run: pip install pyzmq scipy"
    ) from exc

from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# ---------------------------------------------------------------------------
# Fixed configuration (plan.md §2.1, §3.6)
# ---------------------------------------------------------------------------

_DEFAULT_INTENTELE_ROOT = pathlib.Path("/data/XJL/project/intentele")
DEFAULT_DATASET_YAML = (
    _DEFAULT_INTENTELE_ROOT
    / "source/intentele/intentele/tasks/intentele/assets/test_dataset.yaml"
)
ZMQ_PORT = 5556
NUM_FRAMES_PER_PACKET = 5   # sub-frames batched into one ZMQ pose message
JOINT_DOF = 29              # dimension of joint_pos / joint_vel

# SMPL-24 kinematic parents (child index -> parent index; root has -1). Same tree as VR3PtPoseVisualizer.
SMPL_PARENT_INDICES: Tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)

# ---------------------------------------------------------------------------
# Optional: minimal SMPL skeleton viewer (PyVista only; no G1, no VR)
# ---------------------------------------------------------------------------


def _smpl_bone_line_connectivity() -> np.ndarray:
    """VTK line cells: [2, i, j, 2, i, j, ...] for each bone."""
    cells: List[int] = []
    for child_joint in range(1, 24):
        parent_joint = SMPL_PARENT_INDICES[child_joint]
        if parent_joint >= 0:
            cells.extend((2, parent_joint, child_joint))
    return np.asarray(cells, dtype=np.int64)


class SimpleSmplSkeletonWindow:
    """One PyVista window: 23 bone segments between 24 joints (canonical smpl_joints)."""

    def __init__(self) -> None:
        try:
            import pyvista as pv
        except ImportError as error:
            raise ImportError(
                "Install pyvista (and vtk) for --vis-smpl, e.g. pip install pyvista vtk"
            ) from error

        self._pv = pv
        pv.set_plot_theme("dark")
        self.plotter = pv.Plotter(window_size=(960, 720), title="SMPL-24 skeleton")
        self.plotter.set_background("black")
        ground = pv.Plane(i_resolution=8, j_resolution=8, i_size=2.5, j_size=2.5)
        self.plotter.add_mesh(ground, style="wireframe", color="dimgray", opacity=0.4)
        self._line_cells = _smpl_bone_line_connectivity()
        initial_points = np.zeros((24, 3), dtype=np.float64)
        poly = pv.PolyData(initial_points)
        poly.lines = self._line_cells
        self._bone_actor = self.plotter.add_mesh(poly, color="lawngreen", line_width=4)
        self.plotter.add_axes()
        self.plotter.camera_position = [(2.2, -1.8, 1.5), (0.0, 0.0, 0.85), (0.0, 0.0, 1.0)]
        self.plotter.show(interactive_update=True)

    def update_skeleton(self, joint_positions: np.ndarray) -> None:
        if joint_positions.shape != (24, 3):
            raise ValueError(f"expected joint_positions shape (24, 3), got {joint_positions.shape}")
        poly = self._pv.PolyData(np.asarray(joint_positions, dtype=np.float64))
        poly.lines = self._line_cells
        self._bone_actor.GetMapper().SetInputData(poly)

    def refresh(self) -> None:
        if self.plotter is None:
            return
        try:
            self.plotter.update()
        except Exception:
            self.plotter = None

    def close(self) -> None:
        plotter = self.plotter
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass
            self.plotter = None


# ---------------------------------------------------------------------------
# Quaternion helpers
# Internal representation for scipy and for ZMQ ``body_quat_w`` root row: [w, x, y, z].
# scipy.spatial.transform.Rotation uses [x, y, z, w] (scalar last).
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return quaternion_wxyz[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return quaternion_xyzw[..., [3, 0, 1, 2]]


def _rotation_from_wxyz(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(_wxyz_to_xyzw(quaternion_wxyz))


def _root_orientation_disk_to_internal_wxyz(
    root_orientation_disk: np.ndarray,
    disk_component_order: str,
) -> np.ndarray:
    """intentele MotionLibNPZ uses xyzw on disk then WXYZ internally; support both layouts."""
    if disk_component_order == "xyzw":
        return _xyzw_to_wxyz(root_orientation_disk.astype(np.float64)).astype(np.float32)
    if disk_component_order == "wxyz":
        return root_orientation_disk.astype(np.float32)
    raise ValueError(f"disk_component_order must be 'xyzw' or 'wxyz', got {disk_component_order!r}")


def _normalize_quaternion_rows_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    rows = quaternions_wxyz.astype(np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (rows / norms).astype(np.float32)


def _interpolate_quaternion_wxyz_normalized_linear(
    quaternion_wxyz_before: np.ndarray,
    quaternion_wxyz_after: np.ndarray,
    interpolation_alpha: float,
) -> np.ndarray:
    """Blend two WXYZ quaternions with normalized linear interpolation (same idea as Pico pose stream)."""
    quaternion_xyzw_before = _wxyz_to_xyzw(np.asarray(quaternion_wxyz_before, dtype=np.float64).reshape(4))
    quaternion_xyzw_after = _wxyz_to_xyzw(np.asarray(quaternion_wxyz_after, dtype=np.float64).reshape(4))
    dot_product = float(np.dot(quaternion_xyzw_before, quaternion_xyzw_after))
    if dot_product < 0.0:
        quaternion_xyzw_after = -quaternion_xyzw_after
    blended_xyzw = (1.0 - interpolation_alpha) * quaternion_xyzw_before + interpolation_alpha * quaternion_xyzw_after
    norm = float(np.linalg.norm(blended_xyzw))
    if norm > 1e-12:
        blended_xyzw = blended_xyzw / norm
    return _xyzw_to_wxyz(blended_xyzw).astype(np.float32)


def interpolate_motion_clip_sample(
    motion_clip: MotionClip,
    continuous_clip_frame_index: float,
    is_holding_first_frame: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one SMPL sample (smpl_pose, smpl_joints, body_quat_w) for a fractional clip timeline."""
    if is_holding_first_frame:
        # 片段结束后：关节/姿态冻结在第 0 帧（与 plan §9 一致），根朝向保持最后一帧，
        # 避免机器人“跟完动作”后航向被拉回第一帧。
        pose_joints_index = 0
        last_frame_index = motion_clip.num_frames - 1
        return (
            motion_clip.smpl_pose[pose_joints_index].copy(),
            motion_clip.smpl_joints[pose_joints_index].copy(),
            motion_clip.root_quaternion_heading_aligned_wxyz[last_frame_index].copy(),
        )

    number_of_frames = motion_clip.num_frames
    if number_of_frames < 1:
        raise ValueError("motion clip has no frames")

    lower_frame_index = int(np.floor(float(continuous_clip_frame_index)))
    upper_frame_index = int(np.minimum(lower_frame_index + 1, number_of_frames - 1))
    interpolation_alpha = float(continuous_clip_frame_index) - float(lower_frame_index)
    lower_frame_index = int(np.clip(lower_frame_index, 0, number_of_frames - 1))

    smpl_joints_before = motion_clip.smpl_joints[lower_frame_index].astype(np.float32)
    smpl_joints_after = motion_clip.smpl_joints[upper_frame_index].astype(np.float32)
    interpolated_smpl_joints = (1.0 - interpolation_alpha) * smpl_joints_before + interpolation_alpha * smpl_joints_after

    quaternion_before = motion_clip.root_quaternion_heading_aligned_wxyz[lower_frame_index]
    quaternion_after = motion_clip.root_quaternion_heading_aligned_wxyz[upper_frame_index]
    interpolated_body_quat_w = _interpolate_quaternion_wxyz_normalized_linear(
        quaternion_before,
        quaternion_after,
        interpolation_alpha,
    )

    smpl_pose_before = motion_clip.smpl_pose[lower_frame_index].astype(np.float32)
    smpl_pose_after = motion_clip.smpl_pose[upper_frame_index].astype(np.float32)
    interpolated_smpl_pose = (1.0 - interpolation_alpha) * smpl_pose_before + interpolation_alpha * smpl_pose_after

    return interpolated_smpl_pose, interpolated_smpl_joints, interpolated_body_quat_w


# Identity ``heading_delta_quaternion`` (WXYZ): no robot — same as intentele with identity ``delta_q`` only.
IDENTITY_QUATERNION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _apply_heading_alignment_like_intentele_motion_command(
    root_orientation_internal_wxyz: np.ndarray,
    joint_poses_world: np.ndarray,
    heading_delta_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy port of ``MotionCommand._apply_heading_alignment`` (intentele ``commands.py``).

    Returns:
        smpl_joints_body: [T, 24, 3] — root-relative joint vectors in the **aligned** root body frame.
        root_orientation_aligned_wxyz: [T, 4] — ``quat_mul(delta_q, root_orientation)`` per frame, WXYZ.
    """
    num_time_steps = joint_poses_world.shape[0]
    # Single quaternion must be shape (4,) — (1, 4) makes scipy return as_matrix() as (1, 3, 3) and breaks @ (T, 24, 3).
    heading_delta_rotation = _rotation_from_wxyz(
        np.asarray(heading_delta_quaternion_wxyz, dtype=np.float64).reshape(4)
    )
    root_rotation_per_frame = _rotation_from_wxyz(root_orientation_internal_wxyz)
    root_orientation_aligned = heading_delta_rotation * root_rotation_per_frame

    rotation_heading_matrix = heading_delta_rotation.as_matrix()
    joint_world_after_heading = joint_poses_world @ rotation_heading_matrix.T
    joint_relative_to_root_world = joint_world_after_heading - joint_world_after_heading[:, 0:1, :]

    smpl_joints_body = np.zeros((num_time_steps, 24, 3), dtype=np.float32)
    for time_index in range(num_time_steps):
        vectors_root_frame = root_orientation_aligned[time_index].inv().apply(
            joint_relative_to_root_world[time_index]
        )
        smpl_joints_body[time_index] = vectors_root_frame.astype(np.float32)

    root_orientation_aligned_wxyz = _xyzw_to_wxyz(root_orientation_aligned.as_quat()).astype(np.float32)
    return smpl_joints_body, root_orientation_aligned_wxyz


# ---------------------------------------------------------------------------
# Per-clip data container
# ---------------------------------------------------------------------------


class MotionClip:
    """Precomputed, ready-to-send data for one .npz file."""

    def __init__(
        self,
        name: str,
        clip_fps: float,
        smpl_joints: np.ndarray,  # [T, 24, 3]  float32 — intentele body-frame convention
        smpl_pose: np.ndarray,  # [T, 21, 3]  float32
        root_quaternion_heading_aligned_wxyz: np.ndarray,  # [T, 4] float32 — for body_quat_w root row
    ) -> None:
        self.name = name
        self.clip_fps = clip_fps
        self.smpl_joints = smpl_joints
        self.smpl_pose = smpl_pose
        self.root_quaternion_heading_aligned_wxyz = root_quaternion_heading_aligned_wxyz

    @property
    def num_frames(self) -> int:
        return self.smpl_joints.shape[0]


# ---------------------------------------------------------------------------
# Loading and precomputation (intentele MotionCommand + MotionLibNPZ semantics)
# ---------------------------------------------------------------------------


def _precompute_clip(
    npz_path: pathlib.Path,
    root_orientation_disk_component_order: str,
    clip_name: Optional[str] = None,
) -> MotionClip:
    """Load one .npz; build ``smpl_joints`` like intentele ``_apply_heading_alignment`` (identity delta_q)."""
    data = np.load(npz_path, allow_pickle=False)

    for required_key in ("fps", "joint_poses", "root_orientation"):
        if required_key not in data:
            raise ValueError(f"missing required key '{required_key}'")

    clip_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if clip_fps <= 0:
        raise ValueError(f"fps must be > 0, got {clip_fps}")

    joint_poses_world = data["joint_poses"].astype(np.float32)
    root_orientation_disk = data["root_orientation"].astype(np.float32)

    if joint_poses_world.ndim != 3 or joint_poses_world.shape[1] != 24 or joint_poses_world.shape[2] != 3:
        raise ValueError(f"joint_poses must be [T, 24, 3], got {joint_poses_world.shape}")
    if root_orientation_disk.ndim != 2 or root_orientation_disk.shape[1] != 4:
        raise ValueError(f"root_orientation must be [T, 4], got {root_orientation_disk.shape}")
    if joint_poses_world.shape[0] != root_orientation_disk.shape[0]:
        raise ValueError(
            f"joint_poses and root_orientation frame count mismatch: "
            f"{joint_poses_world.shape[0]} vs {root_orientation_disk.shape[0]}"
        )
    if joint_poses_world.shape[0] < 1:
        raise ValueError("clip has zero frames")

    root_orientation_internal_wxyz = _normalize_quaternion_rows_wxyz(
        _root_orientation_disk_to_internal_wxyz(root_orientation_disk, root_orientation_disk_component_order)
    )

    smpl_joints, root_quaternion_heading_aligned_wxyz = _apply_heading_alignment_like_intentele_motion_command(
        root_orientation_internal_wxyz,
        joint_poses_world,
        IDENTITY_QUATERNION_WXYZ,
    )

    smpl_pose = np.zeros((joint_poses_world.shape[0], 21, 3), dtype=np.float32)

    name = clip_name if clip_name is not None else npz_path.stem
    return MotionClip(
        name=name,
        clip_fps=clip_fps,
        smpl_joints=smpl_joints,
        smpl_pose=smpl_pose,
        root_quaternion_heading_aligned_wxyz=root_quaternion_heading_aligned_wxyz,
    )


def load_clips_from_dataset_yaml(
    dataset_yaml: pathlib.Path,
    intentele_root: pathlib.Path,
    root_orientation_disk_component_order: str,
) -> List[MotionClip]:
    """Load npz clips listed in an intentele dataset yaml (e.g. test_dataset.yaml)."""
    with open(dataset_yaml, "r") as yaml_file:
        motion_config = yaml.safe_load(yaml_file)

    motion_root = intentele_root / motion_config["root_path"]
    clips: List[MotionClip] = []
    for motion_entry in motion_config["motions"]:
        if not isinstance(motion_entry, dict) or "file" not in motion_entry:
            continue
        rel = motion_entry["file"]
        if not str(rel).endswith(".npz"):
            continue
        npz_path = motion_root / rel
        try:
            clip = _precompute_clip(
                npz_path,
                root_orientation_disk_component_order=root_orientation_disk_component_order,
                clip_name=rel,
            )
            print(f"  loaded  {rel:<40}  {clip.num_frames} frames @ {clip.clip_fps} fps")
            clips.append(clip)
        except Exception as error:
            print(f"  SKIP    {rel}: {error}")

    if not clips:
        raise RuntimeError(f"No valid clips could be loaded from {dataset_yaml}.")
    return clips


# ---------------------------------------------------------------------------
# zmq_manager command sequencing
# ---------------------------------------------------------------------------


def _activate_control_and_streamed_motion(zmq_socket) -> None:
    """Start policy in PLANNER mode first, then switch to STREAMED_MOTION.

    zmq_manager only consumes ``start=True`` inside ``handlePlannerInput``.
    A single command with ``planner=False`` switches mode in ``update()`` before
    ``handle_input()`` runs, so ``start`` would be dropped.
    """
    zmq_socket.send(build_command_message(start=True, stop=False, planner=True))
    zmq_socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    time.sleep(0.2)
    zmq_socket.send(build_command_message(start=True, stop=False, planner=False))


# ---------------------------------------------------------------------------
# ZMQ packet builder (plan.md §4)
# ---------------------------------------------------------------------------


def _build_pose_packet_from_interpolated_batches(
    smpl_pose_batch: np.ndarray,
    smpl_joints_batch: np.ndarray,
    body_quat_w_batch: np.ndarray,
    global_frame_indices: List[int],
) -> bytes:
    batch_size = int(smpl_pose_batch.shape[0])
    if smpl_joints_batch.shape[0] != batch_size or body_quat_w_batch.shape[0] != batch_size:
        raise ValueError("batch dimension mismatch for interpolated pose arrays")
    if len(global_frame_indices) != batch_size:
        raise ValueError("global_frame_indices length must match batch size")
    pose_data = {
        "smpl_pose": smpl_pose_batch.astype(np.float32, copy=False),
        "smpl_joints": smpl_joints_batch.astype(np.float32, copy=False),
        "body_quat_w": body_quat_w_batch.astype(np.float32, copy=False),
        "joint_pos": np.zeros((batch_size, JOINT_DOF), dtype=np.float32),
        "joint_vel": np.zeros((batch_size, JOINT_DOF), dtype=np.float32),
        "frame_index": np.array(global_frame_indices, dtype=np.int64),
    }
    return pack_pose_message(pose_data, topic="pose", version=3)


# ---------------------------------------------------------------------------
# Non-blocking keyboard reader (separate daemon thread)
# ---------------------------------------------------------------------------


def _start_keyboard_reader() -> queue.SimpleQueue:
    key_queue: queue.SimpleQueue = queue.SimpleQueue()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    # 守护线程退出时 finally 不会执行，用 atexit 保证终端设置始终被还原
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old_settings)

    def _reader() -> None:
        try:
            tty.setcbreak(fd)   # 只禁用 ICANON+ECHO，保留 OPOST，\n 仍正常转 \r\n
            while True:
                character = sys.stdin.read(1)
                if not character:
                    break
                key_queue.put(character)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    threading.Thread(target=_reader, daemon=True).start()
    return key_queue


# ---------------------------------------------------------------------------
# Main player loop  (plan.md §9.4 state machine)
# ---------------------------------------------------------------------------


def run_player(
    clips: List[MotionClip],
    zmq_socket,
    smpl_skeleton_viewer: Optional[SimpleSmplSkeletonWindow] = None,
    visualization_interval_seconds: float = 1.0 / 30.0,
    target_output_frames_per_second: float = 50.0,
) -> None:
    if target_output_frames_per_second <= 0.0:
        raise ValueError("target_output_frames_per_second must be > 0")

    # --- mutable state ---
    current_clip_index = 0
    is_streaming = False
    is_holding_first_frame = False   # True after clip ends; repeat frame 0 (§9)
    global_stream_frame_index = 0
    next_output_deadline_monotonic = time.monotonic()
    previous_clip_index = 0
    last_skeleton_visualization_time = 0.0
    continuous_clip_frame_index = 0.0

    # buffers for building one packet (interpolated rows, not integer clip indices)
    smpl_pose_row_buffer: List[np.ndarray] = []
    smpl_joints_row_buffer: List[np.ndarray] = []
    body_quat_w_row_buffer: List[np.ndarray] = []
    global_frame_buffer: List[int] = []

    clip = clips[current_clip_index]
    output_period_seconds = 1.0 / float(target_output_frames_per_second)

    # 在进入 raw 模式之前完成初始打印，避免 \n 不含 \r 导致的错位
    print(f"\n{len(clips)} clip(s) loaded.  Current: {clip.name}  (clip {clip.clip_fps} fps)")
    print(f"Output stream: {target_output_frames_per_second} Hz (interpolated along clip timeline)")
    print("Keys:  T=play/resume  R=restart  N=next  P=prev  Ctrl-C=quit\n")

    key_queue = _start_keyboard_reader()

    zmq_socket.send(build_command_message(start=True, stop=False, planner=True))
    zmq_socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))

    while True:
        # ---- process all pending keys ----
        should_quit = False
        while True:
            try:
                key = key_queue.get_nowait()
            except queue.Empty:
                break

            if ord(key) == 3:   # Ctrl-C
                should_quit = True
                break

            elif key in ("t", "T"):
                _activate_control_and_streamed_motion(zmq_socket)
                is_streaming = True
                is_holding_first_frame = False
                continuous_clip_frame_index = 0.0
                next_output_deadline_monotonic = time.monotonic()
                print(f"  play  clip={clip.name}  continuous_frame_index=0")

            elif key in ("r", "R"):
                _activate_control_and_streamed_motion(zmq_socket)
                is_streaming = True
                is_holding_first_frame = False
                continuous_clip_frame_index = 0.0
                smpl_pose_row_buffer.clear()
                smpl_joints_row_buffer.clear()
                body_quat_w_row_buffer.clear()
                global_frame_buffer.clear()
                print(f"  restart  clip={clip.name}")

            elif key in ("n", "N"):
                current_clip_index = (current_clip_index + 1) % len(clips)

            elif key in ("p", "P"):
                current_clip_index = (current_clip_index - 1) % len(clips)

        if should_quit:
            break

        # ---- handle clip change (§8.2) ----
        if current_clip_index != previous_clip_index:
            clip = clips[current_clip_index]
            is_holding_first_frame = False
            continuous_clip_frame_index = 0.0
            smpl_pose_row_buffer.clear()
            smpl_joints_row_buffer.clear()
            body_quat_w_row_buffer.clear()
            global_frame_buffer.clear()
            next_output_deadline_monotonic = time.monotonic()
            previous_clip_index = current_clip_index
            print(f"  switched to clip: {clip.name}  ({clip.clip_fps} fps)")

        # ---- sleep until next output tick (uniform target_output_frames_per_second) ----
        if not is_streaming:
            time.sleep(0.02)  # ~50 Hz key polling when idle
            continue

        sleep_seconds = next_output_deadline_monotonic - time.monotonic()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        interpolated_smpl_pose, interpolated_smpl_joints, interpolated_body_quat_w = (
            interpolate_motion_clip_sample(
                motion_clip=clip,
                continuous_clip_frame_index=continuous_clip_frame_index,
                is_holding_first_frame=is_holding_first_frame,
            )
        )

        smpl_pose_row_buffer.append(interpolated_smpl_pose)
        smpl_joints_row_buffer.append(interpolated_smpl_joints)
        body_quat_w_row_buffer.append(interpolated_body_quat_w)
        global_frame_buffer.append(global_stream_frame_index)
        global_stream_frame_index += 1

        if not is_holding_first_frame:
            clip_frame_delta_per_output_tick = float(clip.clip_fps) / float(target_output_frames_per_second)
            continuous_clip_frame_index += clip_frame_delta_per_output_tick
            if continuous_clip_frame_index >= float(clip.num_frames):
                continuous_clip_frame_index = 0.0
                is_holding_first_frame = True
                is_streaming = False
                zmq_socket.send(build_command_message(start=True, stop=False, planner=True))
                zmq_socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
                print(
                    f"  '{clip.name}' ended — holding smpl pose/joints @ frame 0, "
                    f"root orientation @ frame {clip.num_frames - 1}"
                )

        # ---- send packet once buffer is full ----
        if len(global_frame_buffer) >= NUM_FRAMES_PER_PACKET:
            smpl_pose_batch = np.stack(smpl_pose_row_buffer, axis=0)
            smpl_joints_batch = np.stack(smpl_joints_row_buffer, axis=0)
            body_quat_w_batch = np.stack(body_quat_w_row_buffer, axis=0)
            packet = _build_pose_packet_from_interpolated_batches(
                smpl_pose_batch=smpl_pose_batch,
                smpl_joints_batch=smpl_joints_batch,
                body_quat_w_batch=body_quat_w_batch,
                global_frame_indices=global_frame_buffer,
            )
            zmq_socket.send(packet)
            smpl_pose_row_buffer.clear()
            smpl_joints_row_buffer.clear()
            body_quat_w_row_buffer.clear()
            global_frame_buffer.clear()

        # ---- advance deadline (cumulative correction keeps long-run rate near target) ----
        next_output_deadline_monotonic += output_period_seconds

        # ---- optional SMPL skeleton window (throttled; does not affect ZMQ timing) ----
        if smpl_skeleton_viewer is not None:
            now = time.monotonic()
            if now - last_skeleton_visualization_time >= visualization_interval_seconds:
                smpl_skeleton_viewer.update_skeleton(interpolated_smpl_joints)
                smpl_skeleton_viewer.refresh()
                last_skeleton_visualization_time = now


# ---------------------------------------------------------------------------
# Sim elastic band (ROS 2)
# ---------------------------------------------------------------------------


def release_sim_elastic_band(wait_seconds: float = 1.0) -> None:
    """Publish /sim/release_band so MuJoCo drops the robot (same as key 9)."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Empty

    owned_init = not rclpy.ok()
    if owned_init:
        rclpy.init()
    node = Node("amass_release_band")
    publisher = node.create_publisher(Empty, "/sim/release_band", 10)
    time.sleep(0.2)
    publisher.publish(Empty())
    node.destroy_node()
    if owned_init and rclpy.ok():
        rclpy.shutdown()
    print(f"Released sim elastic band, waiting {wait_seconds:.1f} s ...")
    time.sleep(wait_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # 在做任何事之前保存终端设置，并确保 OPOST 开启
    # （上次 tty.setraw 意外退出可能留下 OPOST=0，导致 \n 不含 \r 而输出错位）
    _stdin_fd = sys.stdin.fileno()
    _original_term = termios.tcgetattr(_stdin_fd)
    _fixed_term = list(_original_term)
    _fixed_term[1] |= termios.OPOST   # oflag: 恢复输出处理，使 \n -> \r\n
    termios.tcsetattr(_stdin_fd, termios.TCSANOW, _fixed_term)

    parser = argparse.ArgumentParser(
        description="Stream AMASS-style npz motions over ZMQ (Pico-compatible pose protocol v3).",
    )
    parser.add_argument(
        "--vis-smpl",
        action="store_true",
        help="Open a minimal PyVista window: SMPL-24 skeleton only (no G1, no VR).",
    )
    parser.add_argument(
        "--root-orientation-disk-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help=(
            "Component order of ``root_orientation`` inside the npz on disk. "
            "``xyzw`` matches intentele ``motion_lib_smpl_npz.py``; use ``wxyz`` for scalar-first files."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=50.0,
        help=(
            "Uniform output timeline in Hz (like Pico pose stream target_fps). "
            "Clip content advances in real time: clip_frame_delta = clip_fps / target_fps per tick."
        ),
    )
    parser.add_argument(
        "--dataset-yaml",
        type=pathlib.Path,
        default=DEFAULT_DATASET_YAML,
        help="Intentele dataset yaml listing motion npz files (default: test_dataset.yaml).",
    )
    parser.add_argument(
        "--intentele-root",
        type=pathlib.Path,
        default=_DEFAULT_INTENTELE_ROOT,
        help="Intentele repo root; used to resolve root_path in the dataset yaml.",
    )
    arguments = parser.parse_args()

    print(f"Loading clips from {arguments.dataset_yaml} ...")
    clips = load_clips_from_dataset_yaml(
        arguments.dataset_yaml,
        arguments.intentele_root,
        root_orientation_disk_component_order=arguments.root_orientation_disk_order,
    )

    smpl_skeleton_viewer: Optional[SimpleSmplSkeletonWindow] = None
    if arguments.vis_smpl:
        smpl_skeleton_viewer = SimpleSmplSkeletonWindow()
        print("SMPL skeleton viewer: press T to stream; close window only stops the plot (use Ctrl-C to exit).")

    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://*:{ZMQ_PORT}")
    time.sleep(0.1)  # let subscriber connect before first message
    print(f"ZMQ PUB socket bound on tcp://*:{ZMQ_PORT}")

    try:
        release_sim_elastic_band(wait_seconds=1.0)
    except Exception as error:
        print(f"Could not release elastic band via ROS2: {error}")
        print("  Start sim with --enable-ros2-sim-reset, or press 9 in the sim window.")

    try:
        run_player(
            clips,
            zmq_socket,
            smpl_skeleton_viewer=smpl_skeleton_viewer,
            target_output_frames_per_second=float(arguments.target_fps),
        )
    except KeyboardInterrupt:
        pass
    finally:
        # 无论以何种方式退出都还原终端到运行前的状态
        termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _original_term)
        if smpl_skeleton_viewer is not None:
            smpl_skeleton_viewer.close()
        zmq_socket.close()
        context.term()
        print("\nPlayer stopped.")


if __name__ == "__main__":
    main()
