"""The MOSS-Image runtime; it imports torch, so ``server.bootstrap`` guards the import."""

from .runner import register_moss

__all__ = ["register_moss"]
