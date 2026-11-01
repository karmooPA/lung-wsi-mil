"""Explicit random streams and resumable CPU/CUDA RNG state."""

import importlib.metadata
import os
import platform
import random

import numpy as np
import torch


def configure_runtime(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Experiment device must be CPU, CUDA, or MPS.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "versions": {
            name: str(importlib.metadata.version(name))
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "pandas",
                "scikit-learn",
                "h5py",
                "matplotlib",
            )
        },
        "num_threads": config.num_threads,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
        "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def derived_seed(seed, fold, stream=0):
    return int(np.random.SeedSequence([seed, fold, stream]).generate_state(1)[0])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def capture_rng(generators):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "torch_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "generators": {key: generator.get_state() for key, generator in generators.items()},
    }


def restore_rng(state, generators):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot resume a CUDA random stream without CUDA.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state["torch_mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("Cannot resume an MPS random stream without MPS.")
        torch.mps.set_rng_state(state["torch_mps"])
    for key, generator in generators.items():
        generator.set_state(state["generators"][key])
