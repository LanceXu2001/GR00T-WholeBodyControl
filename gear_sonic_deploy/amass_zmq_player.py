#!/usr/bin/env python3
"""AMASS motion ZMQ player.

Publishes motion clips from gear_sonic_deploy/reference/motions/*.npz
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
  Space    toggle planner / streamed pose (same bytes as Pico A+X on mode switch)
  Ctrl-C   quit

Usage (from repo root):
  python gear_sonic_deploy/amass_zmq_player.py
  python gear_sonic_deploy/amass_zmq_player.py --vis-smpl   # optional SMPL-24 skeleton window
  python gear_sonic_deploy/amass_zmq_player.py --root-orientation-disk-order wxyz   # if your npz stores scalar-first

Default ``--root-orientation-disk-order xyzw`` matches intentele ``motion_lib_smpl_npz.py`` (scalar last on disk).
"""

from __future__ import annotations

import argparse
import pathlib
import queue
import sys
import termios
import threading
import time
import tty
from typing import List, Optional, Tuple

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# ---------------------------------------------------------------------------
# Fixed configuration (plan.md §2.1, §3.6)
# ---------------------------------------------------------------------------

MOTIONS_DIRECTORY = pathlib.Path(__file__).parent / "reference" / "motions"/ "Male2Walking_c3d"
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

    return MotionClip(
        name=npz_path.stem,
        clip_fps=clip_fps,
        smpl_joints=smpl_joints,
        smpl_pose=smpl_pose,
        root_quaternion_heading_aligned_wxyz=root_quaternion_heading_aligned_wxyz,
    )


def load_all_clips(
    motions_directory: pathlib.Path,
    root_orientation_disk_component_order: str,
) -> List[MotionClip]:
    npz_files = sorted(motions_directory.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {motions_directory}")

    clips = []
    for npz_path in npz_files:
        try:
            clip = _precompute_clip(
                npz_path,
                root_orientation_disk_component_order=root_orientation_disk_component_order,
            )
            print(f"  loaded  {npz_path.name:<40}  {clip.num_frames} frames @ {clip.clip_fps} fps")
            clips.append(clip)
        except Exception as error:
            print(f"  SKIP    {npz_path.name}: {error}")

    if not clips:
        raise RuntimeError("No valid clips could be loaded.")
    return clips


# ---------------------------------------------------------------------------
# ZMQ packet builder (plan.md §4)
# ---------------------------------------------------------------------------


def _build_pose_packet(
    clip: MotionClip,
    clip_local_frame_indices: List[int],
    global_frame_indices: List[int],
) -> bytes:
    batch_size = len(clip_local_frame_indices)
    pose_data = {
        "smpl_pose":   clip.smpl_pose[clip_local_frame_indices],                # [B, 21, 3]
        "smpl_joints": clip.smpl_joints[clip_local_frame_indices],              # [B, 24, 3]
        "body_quat_w": clip.root_quaternion_heading_aligned_wxyz[clip_local_frame_indices],  # [B, 4]
        "joint_pos":   np.zeros((batch_size, JOINT_DOF), dtype=np.float32),
        "joint_vel":   np.zeros((batch_size, JOINT_DOF), dtype=np.float32),
        "frame_index": np.array(global_frame_indices, dtype=np.int64),
    }
    return pack_pose_message(pose_data, topic="pose", version=3)


# ---------------------------------------------------------------------------
# Non-blocking keyboard reader (separate daemon thread)
# ---------------------------------------------------------------------------


def _start_keyboard_reader() -> queue.SimpleQueue:
    key_queue: queue.SimpleQueue = queue.SimpleQueue()

    def _reader() -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                character = sys.stdin.read(1)
                if not character:
                    break
                key_queue.put(character)
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
) -> None:
    # --- mutable state ---
    current_clip_index = 0
    current_frame_index = 0
    is_streaming = False
    is_holding_first_frame = False   # True after clip ends; repeat frame 0 (§9)
    global_stream_frame_index = 0
    next_send_time = time.monotonic()
    previous_clip_index = 0
    last_skeleton_visualization_time = 0.0

    # buffers for building one packet
    clip_local_frame_buffer: List[int] = []
    global_frame_buffer: List[int] = []

    key_queue = _start_keyboard_reader()
    clip = clips[current_clip_index]

    print(f"\n{len(clips)} clip(s) loaded.  Current: {clip.name}  ({clip.clip_fps} fps)")
    print("Keys:  T=play/resume  R=restart  N=next  P=prev  Space=planner/pose  Ctrl-C=quit\n")

    # Match pico_manager startup: streamed motion + one planner bootstrap frame.
    receiver_planner_mode = False
    zmq_socket.send(build_command_message(start=False, stop=False, planner=False))
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
                is_streaming = True
                is_holding_first_frame = False
                next_send_time = time.monotonic()
                print(f"  play  clip={clip.name}  frame={current_frame_index}")

            elif key in ("r", "R"):
                current_frame_index = 0
                is_holding_first_frame = False
                clip_local_frame_buffer.clear()
                global_frame_buffer.clear()
                print(f"  restart  clip={clip.name}")

            elif key in ("n", "N"):
                current_clip_index = (current_clip_index + 1) % len(clips)

            elif key in ("p", "P"):
                current_clip_index = (current_clip_index - 1) % len(clips)

            elif key == " ":
                # Same as pico_manager A+X on edge: PLANNER<->POSE uses start=True, stop=False
                # (see pico_manager_thread_server.py around socket.send(build_command_message(...))).
                receiver_planner_mode = not receiver_planner_mode
                zmq_socket.send(
                    build_command_message(
                        start=True,
                        stop=False,
                        planner=receiver_planner_mode,
                    )
                )
                if receiver_planner_mode:
                    zmq_socket.send(
                        build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0)
                    )
                mode_label = "planner" if receiver_planner_mode else "streamed pose (npz)"
                print(f"  command toggle -> {mode_label}")

        if should_quit:
            break

        # ---- handle clip change (§8.2) ----
        if current_clip_index != previous_clip_index:
            clip = clips[current_clip_index]
            current_frame_index = 0
            is_holding_first_frame = False
            clip_local_frame_buffer.clear()
            global_frame_buffer.clear()
            next_send_time = time.monotonic()
            previous_clip_index = current_clip_index
            print(f"  switched to clip: {clip.name}  ({clip.clip_fps} fps)")

        # ---- sleep until next sub-frame deadline (§3.6) ----
        if not is_streaming:
            time.sleep(0.02)  # ~50 Hz key polling when idle
            continue

        sleep_seconds = next_send_time - time.monotonic()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        # ---- choose clip-local frame for this step ----
        if is_holding_first_frame:
            frame_for_this_step = 0
        else:
            frame_for_this_step = current_frame_index

        clip_local_frame_buffer.append(frame_for_this_step)
        global_frame_buffer.append(global_stream_frame_index)
        global_stream_frame_index += 1

        # advance within clip unless holding at frame 0
        if not is_holding_first_frame:
            current_frame_index += 1
            if current_frame_index >= clip.num_frames:
                current_frame_index = 0
                is_holding_first_frame = True
                print(f"  '{clip.name}' ended — holding frame 0")

        # ---- send packet once buffer is full ----
        if len(clip_local_frame_buffer) >= NUM_FRAMES_PER_PACKET:
            packet = _build_pose_packet(clip, clip_local_frame_buffer, global_frame_buffer)
            zmq_socket.send(packet)
            clip_local_frame_buffer.clear()
            global_frame_buffer.clear()

        # ---- advance deadline by one sub-frame (cumulative-correction scheduling) ----
        next_send_time += 1.0 / clip.clip_fps

        # ---- optional SMPL skeleton window (throttled; does not affect ZMQ timing) ----
        if smpl_skeleton_viewer is not None:
            now = time.monotonic()
            if now - last_skeleton_visualization_time >= visualization_interval_seconds:
                smpl_skeleton_viewer.update_skeleton(clip.smpl_joints[frame_for_this_step])
                smpl_skeleton_viewer.refresh()
                last_skeleton_visualization_time = now


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
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
    arguments = parser.parse_args()

    print(f"Scanning {MOTIONS_DIRECTORY} ...")
    clips = load_all_clips(
        MOTIONS_DIRECTORY,
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
        run_player(clips, zmq_socket, smpl_skeleton_viewer=smpl_skeleton_viewer)
    except KeyboardInterrupt:
        pass
    finally:
        if smpl_skeleton_viewer is not None:
            smpl_skeleton_viewer.close()
        zmq_socket.close()
        context.term()
        print("\nPlayer stopped.")


if __name__ == "__main__":
    main()
