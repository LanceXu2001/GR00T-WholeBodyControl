"""Wall-clock ZMQ pose publisher that also writes reference motion CSV."""

import csv
import threading
import time
from pathlib import Path
from typing import Callable, List

import numpy as np


class ReferencePublisher:
    """Publishes one MotionClip via ZMQ at target_fps and records smpl_joints to CSV.

    Usage:
        publisher.start(trial_dir)   # starts background thread
        publisher.clip_finished.wait()  # or check is_set()
        publisher.stop()             # joins thread, flushes CSV
    """

    FRAMES_PER_ZMQ_PACKET = 5

    def __init__(self, motion_clip, target_fps: float, zmq_socket, build_pose_packet: Callable):
        self._motion_clip = motion_clip
        self._target_fps = target_fps
        self._zmq_socket = zmq_socket
        self._build_pose_packet = build_pose_packet

        self.clip_finished = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None
        self._csv_file = None
        self._csv_writer = None

    def start(self, trial_dir: str) -> None:
        self.clip_finished.clear()
        self._stop_event.clear()

        reference_dir = Path(trial_dir) / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(reference_dir / "ref_motion.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        joint_columns = [f"joint_{j}_{axis}" for j in range(24) for axis in ("x", "y", "z")]
        self._csv_writer.writerow(["frame_index", "wall_time"] + joint_columns)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self._csv_file.close()

    def _run(self) -> None:
        from amass_zmq_player import interpolate_motion_clip_sample

        clip = self._motion_clip
        output_period_seconds = 1.0 / self._target_fps
        continuous_clip_frame = 0.0
        global_frame_index = 0
        is_holding_last_frame = False

        smpl_pose_buffer: List[np.ndarray] = []
        smpl_joints_buffer: List[np.ndarray] = []
        body_quat_buffer: List[np.ndarray] = []
        frame_index_buffer: List[int] = []

        next_deadline = time.monotonic()

        while not self._stop_event.is_set():
            sleep_seconds = next_deadline - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            smpl_pose, smpl_joints, body_quat_w = interpolate_motion_clip_sample(
                clip, continuous_clip_frame, is_holding_last_frame
            )

            smpl_pose_buffer.append(smpl_pose)
            smpl_joints_buffer.append(smpl_joints)
            body_quat_buffer.append(body_quat_w)
            frame_index_buffer.append(global_frame_index)
            self._csv_writer.writerow(
                [global_frame_index, time.time()] + smpl_joints.flatten().tolist()
            )

            if not is_holding_last_frame:
                continuous_clip_frame += clip.clip_fps / self._target_fps
                if continuous_clip_frame >= clip.num_frames:
                    continuous_clip_frame = 0.0
                    is_holding_last_frame = True
                    self.clip_finished.set()

            if len(frame_index_buffer) >= self.FRAMES_PER_ZMQ_PACKET:
                self._zmq_socket.send(
                    self._build_pose_packet(
                        smpl_pose_batch=np.stack(smpl_pose_buffer),
                        smpl_joints_batch=np.stack(smpl_joints_buffer),
                        body_quat_w_batch=np.stack(body_quat_buffer),
                        global_frame_indices=frame_index_buffer,
                    )
                )
                smpl_pose_buffer.clear()
                smpl_joints_buffer.clear()
                body_quat_buffer.clear()
                frame_index_buffer.clear()

            global_frame_index += 1
            next_deadline += output_period_seconds
