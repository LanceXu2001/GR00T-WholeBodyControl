#!/usr/bin/env python3
"""AMASS motion ZMQ player.

Publishes motion clips from gear_sonic_deploy/reference/motions/*.npz
as a Pico-compatible ZMQ stream (protocol version 3, PUB socket port 5556).

Required npz keys (written-once convention, §3.0 of plan.md):
  fps               scalar > 0          clip frame rate (Hz)
  joint_poses       float32 [T, 24, 3]  world-space joint positions
  root_orientation  float32 [T, 4]      world-space root quaternion [w, x, y, z]

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
  python gear_sonic_deploy/amass_zmq_player.py --yaw-degrees 180   # optional: flip horizontal facing (+Z axis, degrees)

The first-frame inverse rotation (plan §3.1) zeros **relative** root orientation at frame 0; it does **not**
guarantee that the skeleton faces world +X. If your clip faces -X in the viewer, try ``--yaw-degrees 180``.
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
ROOT_JOINT_INDEX = 0        # pelvis is joint index 0 inside joint_poses

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
# Quaternion convention: [w, x, y, z] throughout this file.
# scipy.spatial.transform.Rotation uses [x, y, z, w] (scalar last).
# Reordering happens in exactly two helper functions below — nowhere else.
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return quaternion_wxyz[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return quaternion_xyzw[..., [3, 0, 1, 2]]


def _rotation_from_wxyz(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(_wxyz_to_xyzw(quaternion_wxyz))


# ---------------------------------------------------------------------------
# Per-clip data container
# ---------------------------------------------------------------------------


class MotionClip:
    """Precomputed, ready-to-send data for one .npz file."""

    def __init__(
        self,
        name: str,
        clip_fps: float,
        smpl_joints: np.ndarray,               # [T, 24, 3]  float32
        smpl_pose: np.ndarray,                 # [T, 21, 3]  float32
        root_quaternion_canonical: np.ndarray, # [T,  4]     float32  wxyz
    ) -> None:
        self.name = name
        self.clip_fps = clip_fps
        self.smpl_joints = smpl_joints
        self.smpl_pose = smpl_pose
        self.root_quaternion_canonical = root_quaternion_canonical

    @property
    def num_frames(self) -> int:
        return self.smpl_joints.shape[0]


# ---------------------------------------------------------------------------
# Loading and precomputation (plan.md §3.0 – §3.5)
# ---------------------------------------------------------------------------


def _apply_extra_world_rotation_about_z(
    smpl_joints: np.ndarray,
    root_quaternion_canonical_wxyz: np.ndarray,
    extra_world_rotation: Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same fixed world rotation to joint offsets and root quaternions (left factor)."""
    extra_matrix = extra_world_rotation.as_matrix()
    joints_out = (smpl_joints @ extra_matrix.T).astype(np.float32)
    canonical_rotations = Rotation.from_quat(_wxyz_to_xyzw(root_quaternion_canonical_wxyz))
    combined = extra_world_rotation * canonical_rotations
    quat_out = _xyzw_to_wxyz(combined.as_quat()).astype(np.float32)
    return joints_out, quat_out


def _precompute_clip(npz_path: pathlib.Path, extra_world_rotation: Optional[Rotation] = None) -> MotionClip:
    """Load one .npz, validate, precompute canonical smpl_joints and root quaternions."""
    data = np.load(npz_path, allow_pickle=False)

    for required_key in ("fps", "joint_poses", "root_orientation"):
        if required_key not in data:
            raise ValueError(f"missing required key '{required_key}'")

    clip_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if clip_fps <= 0:
        raise ValueError(f"fps must be > 0, got {clip_fps}")

    joint_poses = data["joint_poses"].astype(np.float32)           # [T, 24, 3]
    root_orientation = data["root_orientation"].astype(np.float32) # [T,  4] wxyz

    if joint_poses.ndim != 3 or joint_poses.shape[1] != 24 or joint_poses.shape[2] != 3:
        raise ValueError(f"joint_poses must be [T, 24, 3], got {joint_poses.shape}")
    if root_orientation.ndim != 2 or root_orientation.shape[1] != 4:
        raise ValueError(f"root_orientation must be [T, 4], got {root_orientation.shape}")
    if joint_poses.shape[0] != root_orientation.shape[0]:
        raise ValueError(
            f"joint_poses and root_orientation frame count mismatch: "
            f"{joint_poses.shape[0]} vs {root_orientation.shape[0]}"
        )
    if joint_poses.shape[0] < 1:
        raise ValueError("clip has zero frames")

    # §3.1 step 1: first-frame root rotation inverse
    first_frame_rotation = _rotation_from_wxyz(root_orientation[0])
    first_frame_rotation_inverse = first_frame_rotation.inv()

    # §3.1 step 2: canonical root quaternion sequence
    # q_canonical[t] = normalize( q_first_inverse ⊗ q_raw[t] )
    # scipy broadcasts vectorised multiplication correctly; normalisation is automatic.
    all_frame_rotations = _rotation_from_wxyz(root_orientation)  # vectorised, shape (T,)
    canonical_rotations = first_frame_rotation_inverse * all_frame_rotations
    root_quaternion_canonical = _xyzw_to_wxyz(canonical_rotations.as_quat()).astype(np.float32)

    # §3.1 step 3: canonical joint positions
    # Order: (a) rotate all joints around world origin with R_first_inverse,
    #        (b) subtract the rotated root position.
    # Using row-vector convention: R @ v  ==  v @ R.T
    rotation_matrix_inverse = first_frame_rotation_inverse.as_matrix()  # [3, 3]
    joint_positions_rotated = joint_poses @ rotation_matrix_inverse.T   # [T, 24, 3]
    root_position_rotated = joint_positions_rotated[:, ROOT_JOINT_INDEX : ROOT_JOINT_INDEX + 1, :]  # [T, 1, 3]
    smpl_joints = (joint_positions_rotated - root_position_rotated).astype(np.float32)

    # §3.2: smpl_pose — zeros (npz minimum set does not include it)
    smpl_pose = np.zeros((joint_poses.shape[0], 21, 3), dtype=np.float32)

    if extra_world_rotation is not None:
        smpl_joints, root_quaternion_canonical = _apply_extra_world_rotation_about_z(
            smpl_joints, root_quaternion_canonical, extra_world_rotation
        )

    return MotionClip(
        name=npz_path.stem,
        clip_fps=clip_fps,
        smpl_joints=smpl_joints,
        smpl_pose=smpl_pose,
        root_quaternion_canonical=root_quaternion_canonical,
    )


def load_all_clips(
    motions_directory: pathlib.Path,
    extra_world_rotation: Optional[Rotation] = None,
) -> List[MotionClip]:
    npz_files = sorted(motions_directory.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {motions_directory}")

    clips = []
    for npz_path in npz_files:
        try:
            clip = _precompute_clip(npz_path, extra_world_rotation=extra_world_rotation)
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
        "body_quat_w": clip.root_quaternion_canonical[clip_local_frame_indices],# [B,  4]
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
        "--yaw-degrees",
        type=float,
        default=0.0,
        help=(
            "After §3.1 canonicalization, rotate the whole clip by this yaw about world +Z (degrees). "
            "Example: 180 if the character faces -X but you want +X in the viewer / deploy frame."
        ),
    )
    arguments = parser.parse_args()

    extra_world_rotation: Optional[Rotation] = None
    if abs(arguments.yaw_degrees) > 1e-9:
        extra_world_rotation = Rotation.from_euler("z", float(np.deg2rad(arguments.yaw_degrees)))

    print(f"Scanning {MOTIONS_DIRECTORY} ...")
    clips = load_all_clips(MOTIONS_DIRECTORY, extra_world_rotation=extra_world_rotation)
    if extra_world_rotation is not None:
        print(f"  extra world yaw about +Z: {arguments.yaw_degrees:g} deg")

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
