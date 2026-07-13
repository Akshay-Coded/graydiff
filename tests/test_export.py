import pytest
import torch

from graydiff.model import Surrogate

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from graydiff.export import export_to_onnx, verify_onnx_parity  # noqa: E402


@pytest.mark.slow
def test_export_and_parity(tmp_path):
    model = Surrogate(hidden=8)
    path = export_to_onnx(model, tmp_path / "surrogate.onnx", grid_size=16)
    assert path.exists()

    max_diff = verify_onnx_parity(model, path, grid_size=16, n_trials=3)
    assert max_diff < 1e-4


@pytest.mark.slow
def test_export_single_file_no_external_data(tmp_path):
    model = Surrogate(hidden=8)
    path = export_to_onnx(model, tmp_path / "surrogate.onnx", grid_size=16)
    files = list(tmp_path.iterdir())
    assert files == [path], f"expected exactly one file, got {files}"


@pytest.mark.slow
def test_export_from_mps_trained_model_still_works(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    model = Surrogate(hidden=8).to("mps")
    path = export_to_onnx(model, tmp_path / "surrogate.onnx", grid_size=16)
    assert path.exists()
    max_diff = verify_onnx_parity(model, path, grid_size=16, n_trials=2)
    assert max_diff < 1e-4
