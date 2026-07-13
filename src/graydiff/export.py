"""Phase 6: export the forward surrogate to ONNX for onnxruntime-web.

Always export from a CPU-resident deep copy of the model, regardless of the
device it was trained/loaded on: torch.onnx.export has historically not
supported exporting directly from MPS tensors, and CPU is the right export
target anyway — the browser's onnxruntime-web runs as WebAssembly/WebGL,
never literally "MPS".
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    path: str | Path,
    grid_size: int = 64,
    opset: int = 18,
) -> Path:
    """Export `model` (a graydiff.model.Surrogate) to ONNX at `path`.

    Uses a fixed input shape [1, 4, grid_size, grid_size] (no dynamic axes)
    deliberately: the forward playground always runs at one grid size, and a
    fixed shape keeps the exported graph simple and keeps CircularConv2d's
    explicit F.pad(mode='circular') call exportable without shape-inference
    ambiguity.

    `external_data=False`: torch's exporter defaults to writing weights to a
    SEPARATE `<path>.data` file (a feature meant for multi-gigabyte models
    that exceed protobuf's 2GB inline limit). This model is ~300KB total, so
    that split only adds deployment friction — the static frontend has to
    fetch and wire up two files instead of one. Force everything into the
    single .onnx file instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(1, 4, grid_size, grid_size, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        external_data=False,
    )
    return path


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: str | Path,
    grid_size: int = 64,
    n_trials: int = 5,
    atol: float = 1e-4,
    seed: int = 0,
) -> float:
    """Compare ONNX Runtime output against the torch-CPU model on random
    inputs. Returns the max absolute difference observed; raises if it
    exceeds atol. The whole point of exporting is that the browser's copy
    of the model behaves identically to the one every other notebook
    validated — this is the check that confirms it actually does.
    """
    import onnxruntime as ort

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = torch.Generator().manual_seed(seed)
    max_diff = 0.0
    for _ in range(n_trials):
        x = torch.randn(1, 4, grid_size, grid_size, dtype=torch.float32, generator=rng)
        with torch.no_grad():
            torch_out = model_cpu(x).numpy()
        onnx_out = session.run(None, {"input": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    if max_diff > atol:
        raise AssertionError(f"ONNX/torch parity check failed: max diff {max_diff} > atol {atol}")
    return max_diff
