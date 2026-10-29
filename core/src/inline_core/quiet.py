"""Silence third-party import-time warnings that are noise rather than signal."""

from __future__ import annotations

import logging


def quiet_dependency_noise() -> None:
    """Called by every entry point, not just the server: both lines reached a user's bug report."""
    for name in ("diffusers.quantizers.torchao.torchao_quantizer", "torch.utils._pytree"):
        logging.getLogger(name).setLevel(logging.ERROR)
