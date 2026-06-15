import torch
import pytest
import numpy as np
import os
import random

from Recurrent_Layers.LSTMTriDim import LSTMTriDimFunct


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    model.double()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@pytest.mark.order(1)
def test_lstm_forward() -> None:
    # define inputs
    inputs = torch.rand(64, 12, 20).double()
    h_0 = torch.rand(1, 64, 30).double()
    c_0 = torch.rand(1, 64, 30).double()

    inputs_torch = inputs.clone()
    h_0_torch = h_0.clone()
    c_0_torch = c_0.clone()

    # define models
    set_seed(42)
    model = LSTMTriDimFunct(20, 30)
    parameters_to_double(model)

    set_seed(42)
    model_torch = torch.nn.LSTM(20, 30, batch_first=True)
    parameters_to_double(model_torch)

    # compute outputs
    outputs, (h_n, c_n) = model(inputs, h_0, c_0)
    outputs_torch, (h_n_torch, c_n_torch) = model_torch(inputs_torch, (h_0_torch, c_0_torch))

    # check output types
    assert isinstance(outputs, torch.Tensor), "Output should be a torch.Tensor"
    assert isinstance(h_n, torch.Tensor), "h_n should be a torch.Tensor"
    assert isinstance(c_n, torch.Tensor), "c_n should be a torch.Tensor"

    # check shapes
    assert outputs.shape == outputs_torch.shape, f"Output shape mismatch: got {outputs.shape}, expected {outputs_torch.shape}"
    assert h_n.shape == h_n_torch.shape, f"h_n shape mismatch: got {h_n.shape}, expected {h_n_torch.shape}"
    assert c_n.shape == c_n_torch.shape, f"c_n shape mismatch: got {c_n.shape}, expected {c_n_torch.shape}"

    # check values
    assert (outputs.round(2) == outputs_torch.round(2)).all(), "Mismatch in outputs"
    assert (h_n.round(2) == h_n_torch.round(2)).all(), "Mismatch in h_n"
    assert (c_n.round(2) == c_n_torch.round(2)).all(), "Mismatch in c_n"

    return None