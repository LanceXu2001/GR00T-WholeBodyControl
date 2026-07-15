"""Configuration dataclass and YAML loader for batch motion collection."""

from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class Config:
    output_root_dir: str
    npz_path: str
    target_fps: float
    num_trials: int
    trial_timeout_sec: float
    fall_height_threshold: float
    ref_idle_sec: float
    start_sim: bool
    start_deploy: bool
    sim_args: List[str]
    deploy_args: List[str]


def load_config(yaml_path: str) -> Config:
    with open(yaml_path) as config_file:
        data = yaml.safe_load(config_file)
    processes = data.get("processes", {})
    return Config(
        output_root_dir=data["output"]["root_dir"],
        npz_path=data["reference"]["npz_path"],
        target_fps=float(data["reference"]["target_fps"]),
        num_trials=int(data["collection"]["num_trials"]),
        trial_timeout_sec=float(data["collection"]["trial_timeout_sec"]),
        fall_height_threshold=float(data["stop_conditions"]["fall_height_threshold"]),
        ref_idle_sec=float(data["stop_conditions"]["ref_idle_sec"]),
        start_sim=bool(processes.get("start_sim", False)),
        start_deploy=bool(processes.get("start_deploy", False)),
        sim_args=list(processes.get("sim_args", [])),
        deploy_args=list(processes.get("deploy_args", [])),
    )
