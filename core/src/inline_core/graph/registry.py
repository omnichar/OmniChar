"""The node registry: descriptors served at /v1/models plus the runner behind each type."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from ..errors import UnknownNodeType
from .descriptor import NodeDescriptor
from .loader_runners import register_loaders
from .primitives import register_primitives
from .runners import (
    AUDIO_INPUT,
    IMAGE_INPUT,
    TEXT_INPUT,
    VIDEO_INPUT,
    AudioInputRunner,
    ImageInputRunner,
    NodeRunner,
    TextInputRunner,
    VideoInputRunner,
)


class Registry:
    def __init__(self) -> None:
        self._descriptors: dict[str, NodeDescriptor] = {}
        self._runners: dict[str, NodeRunner] = {}

    def register(self, descriptor: NodeDescriptor, runner: NodeRunner | None = None) -> None:
        """Register a node. A descriptor with no runner is served + validated but cannot run yet."""
        self._descriptors[descriptor.type] = descriptor
        if runner is not None:
            self._runners[descriptor.type] = runner

    def unregister(self, node_type: str) -> None:
        """Drop a node type. Used when an extension is disabled; unknown types are ignored.

        Only the registry entry goes away - a module already imported stays in ``sys.modules``
        (Python cannot unload), but nothing can reach it once no descriptor points at its runner."""
        self._descriptors.pop(node_type, None)
        self._runners.pop(node_type, None)

    def get(self, node_type: str) -> NodeDescriptor:
        descriptor = self._descriptors.get(node_type)
        if descriptor is None:
            raise UnknownNodeType(f"Unknown node type {node_type!r}.")
        return descriptor

    def has(self, node_type: str) -> bool:
        return node_type in self._descriptors

    def has_runner(self, node_type: str) -> bool:
        """Whether this type can execute. A descriptor may be served with no runner behind it."""
        return node_type in self._runners

    def runner(self, node_type: str) -> NodeRunner:
        runner = self._runners.get(node_type)
        if runner is None:
            raise UnknownNodeType(f"No runner registered for {node_type!r}.")
        return runner

    def descriptors(self) -> list[NodeDescriptor]:
        return list(self._descriptors.values())

    def version(self) -> str:
        """A fingerprint of the **full descriptor content**, not just the set of node types.

        Node types alone are not enough now that extensions can be toggled and switched between
        versions: an upgrade that changes a param default while keeping its node types would
        produce an identical fingerprint, and the client - which caches ``/v1/models`` against this
        as an ETag - would keep serving the stale catalog.
        """
        payload = json.dumps(
            [asdict(d) for _, d in sorted(self._descriptors.items())],
            separators=(",", ":"),
            sort_keys=True,
            default=str,  # enum members serialize by value; anything exotic by repr
        )
        return f"r_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def build_default_registry() -> Registry:
    """A registry with the built-in source nodes, the loader subnodes, and the remaining primitive
    descriptors.

    Source nodes have runners but stay hidden (the Studio drives text/image inputs through its own
    Prompt/library nodes). The ``load/*`` nodes now have runners and are **visible** - they feed a
    model node's component inputs. The rest of the primitives (encode/sample/decode) are
    descriptor-only and hidden until their runners land (C2).
    """
    registry = Registry()
    registry.register(replace(TEXT_INPUT, hidden=True), TextInputRunner())
    registry.register(replace(IMAGE_INPUT, hidden=True), ImageInputRunner())
    registry.register(replace(VIDEO_INPUT, hidden=True), VideoInputRunner())
    registry.register(replace(AUDIO_INPUT, hidden=True), AudioInputRunner())
    register_primitives(registry)
    register_loaders(registry)
    return registry
