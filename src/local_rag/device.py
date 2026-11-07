"""One place that chooses the compute device: CUDA first, CPU as fallback."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_device(setting: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. 'auto' picks CUDA when available. The choice is logged."""
    import torch

    setting = (setting or "auto").lower()
    if setting not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, got {setting!r}")

    cuda_ok = torch.cuda.is_available()
    if setting == "cpu":
        device = "cpu"
    elif setting == "cuda":
        if not cuda_ok:
            log.warning("device=cuda requested but CUDA is not available, falling back to CPU")
        device = "cuda" if cuda_ok else "cpu"
    else:
        device = "cuda" if cuda_ok else "cpu"

    if device == "cuda":
        log.info("Device: cuda (%s, torch %s, CUDA %s)",
                 torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
    else:
        log.info("Device: cpu (torch %s)", torch.__version__)
    return device
