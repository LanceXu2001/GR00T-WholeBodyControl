#!/usr/bin/env python3
"""Automated batch motion data collection.

Runs N trials of: reset → publish reference → record ROS2 data → stop.
Requires sim with --enable-ros2-body-state and deploy with
HAS_ROS2=1 and --input-type zmq_manager.

Usage:
    python gear_sonic_deploy/scripts/batch_motion_collect.py \
        --config gear_sonic_deploy/config/motion_collect_example.yaml
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import zmq

# Make gear_sonic_deploy/ and repo root importable
_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _DEPLOY_ROOT.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (_DEPLOY_ROOT, _REPO_ROOT, _SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amass_zmq_player import _build_pose_packet_from_interpolated_batches, _precompute_clip
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message
from motion_collect.config import load_config
from motion_collect.reference_publisher import ReferencePublisher
from motion_collect.ros2_collector import Ros2Collector


def activate_streamed_motion(zmq_socket: zmq.Socket) -> None:
    """Start deploy CONTROL and switch to STREAMED_MOTION (never send stop=True).

    deploy defaults to PLANNER mode; operator_state.start is set during planner
    init. Then planner=False switches to streamed pose + enables ZMQ streaming.
    """
    zmq_socket.send(build_command_message(start=True, stop=False, planner=True))
    time.sleep(0.5)
    for _ in range(10):
        zmq_socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        time.sleep(0.05)
    time.sleep(0.5)
    zmq_socket.send(build_command_message(start=True, stop=False, planner=False))
    time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch motion data collection")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.output_root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally start sim and deploy as subprocesses
    sim_process = None
    deploy_process = None
    if config.start_sim:
        sim_process = subprocess.Popen(
            ["python", str(_REPO_ROOT / "gear_sonic/scripts/run_sim_loop.py")] + config.sim_args
        )
    if config.start_deploy:
        deploy_process = subprocess.Popen(
            ["bash", str(_DEPLOY_ROOT / "deploy.sh"), "sim"] + config.deploy_args
        )
    if sim_process or deploy_process:
        print("Waiting 5 s for sim/deploy to initialise ...")
        time.sleep(5.0)

    # ZMQ publisher socket (same port as amass_zmq_player)
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind("tcp://*:5556")
    time.sleep(0.5)  # give deploy time to connect

    # Load motion clip (default disk quaternion order matches amass_zmq_player)
    motion_clip = _precompute_clip(Path(config.npz_path), root_orientation_disk_component_order="xyzw")
    print(f"Loaded clip '{motion_clip.name}': {motion_clip.num_frames} frames @ {motion_clip.clip_fps} fps")

    reference_publisher = ReferencePublisher(
        motion_clip=motion_clip,
        target_fps=config.target_fps,
        zmq_socket=zmq_socket,
        build_pose_packet=_build_pose_packet_from_interpolated_batches,
    )

    ros2_collector = Ros2Collector()
    spin_thread = threading.Thread(target=ros2_collector.spin_forever, daemon=True)
    spin_thread.start()

    print("Waiting for /sim/bodies/names ...")
    ros2_collector.wait_for_body_names(timeout=10.0)
    print(f"  {len(ros2_collector.body_names)} bodies: {ros2_collector.body_names[:5]} ...")

    print("Releasing sim elastic band, waiting 1 s ...")
    ros2_collector.publish_release_band()
    time.sleep(1.0)

    print("Activating deploy CONTROL + STREAMED_MOTION ...")
    activate_streamed_motion(zmq_socket)

    manifest = {
        "total_trials": config.num_trials,
        "completed_trials": 0,
        "npz_path": config.npz_path,
        "trials": [],
    }

    for trial_id in range(config.num_trials):
        trial_dir = output_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir()
        print(f"\ntrial {trial_id + 1}/{config.num_trials} → {trial_dir}")

        if trial_id > 0:
            print("  reset sim + deploy ...")
            ros2_collector.publish_sim_reset()
            time.sleep(0.5)
            activate_streamed_motion(zmq_socket)

        # Start recording and reference publishing simultaneously
        ros2_collector.start_trial(str(trial_dir))
        reference_publisher.start(str(trial_dir))

        # Poll stop conditions
        trial_start_time = time.monotonic()
        end_reason = "timeout"
        while True:
            elapsed = time.monotonic() - trial_start_time
            if elapsed >= config.trial_timeout_sec:
                end_reason = "timeout"
                break
            if ros2_collector.pelvis_z < config.fall_height_threshold:
                end_reason = "fallen"
                break
            if reference_publisher.clip_finished.is_set():
                time.sleep(config.ref_idle_sec)  # let controller finish the motion
                end_reason = "reference_finished"
                break
            time.sleep(0.05)

        wall_duration = time.monotonic() - trial_start_time

        # Stop reference and recording (do not send stop=True — it shuts down deploy)
        reference_publisher.stop()
        ros2_collector.stop_trial()

        print(f"  → {end_reason} ({wall_duration:.1f} s)")

        trial_meta = {
            "trial_id": trial_id,
            "end_reason": end_reason,
            "wall_duration_sec": round(wall_duration, 3),
        }
        (trial_dir / "trial_meta.json").write_text(json.dumps(trial_meta, indent=2))

        manifest["trials"].append({"id": trial_id, "end_reason": end_reason, "wall_sec": round(wall_duration, 2)})
        manifest["completed_trials"] = trial_id + 1
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Cleanup
    ros2_collector.shutdown()
    zmq_socket.close()
    zmq_context.term()
    if sim_process:
        sim_process.terminate()
    if deploy_process:
        deploy_process.terminate()


if __name__ == "__main__":
    main()
