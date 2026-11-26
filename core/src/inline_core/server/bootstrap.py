"""Best-effort model registration. A model whose deps are absent is skipped; the engine still serves
source nodes and the rest of the API, so a torch-less install boots cleanly.

Built-ins register first and community extensions last: ``Registry.register`` replaces by node type,
so the reverse order would let an installed extension silently take over ``alibaba/z-image-turbo``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..device.policy import DevicePolicy
from ..graph.registry import Registry
from ..models.requirements import RequirementsRegistry
from ..runtime.store import TakeStore

if TYPE_CHECKING:
    from ..extensions.loader import LoadedExtension


def register_models(
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    *,
    requirements: RequirementsRegistry | None = None,
    rpc: Any = None,
    events: Any = None,
) -> tuple[list[str], list[LoadedExtension]]:
    """Register the built-in models, then load installed extensions.

    Returns ``(built_in_node_types, loaded_extensions)``. Neither half can take the server down: a
    built-in whose deps are missing is skipped, and an extension that raises is reported on its
    ``LoadedExtension`` for the Extensions UI.
    """
    reqs = requirements if requirements is not None else RequirementsRegistry()
    registered = _register_builtins(registry, store, policy, reqs)
    extensions = _load_extensions(registry, store, policy, reqs, rpc, events)
    return registered, extensions


def _register_builtins(
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    requirements: RequirementsRegistry,
) -> list[str]:
    registered: list[str] = []
    # Torch-free: the Control Space node is client-side, so its ControlNet download suggestion is
    # offered even on a runtime-less install (registered unconditionally, unlike the model runners).
    from ..models.controlspace import ControlSpaceProvider

    requirements.register("controlSpace", ControlSpaceProvider())
    from ..models.characterreqs import ENCODER_NODES, CharacterEncoderProvider

    for node_type in ENCODER_NODES:
        requirements.register(node_type, CharacterEncoderProvider())
    from ..models.trainingreqs import TRAINING_NODES, TrainingBaseProvider

    for node_type in TRAINING_NODES:
        requirements.register(node_type, TrainingBaseProvider())
    registered.append("controlSpace")
    try:
        from ..models.character import register_character_nodes

        register_character_nodes(registry)
        registered.append("character")
    except ImportError:
        pass
    try:
        from ..models.zimage.provider import ZImageProvider
        from ..models.zimage.runner import register_zimage

        register_zimage(registry, store, policy)
        # Torch-free, but guarded by the same import: with no runtime the node is unavailable, so
        # advertising its requirements would offer a download for a node that cannot run.
        requirements.register("alibaba/z-image-turbo", ZImageProvider())
        registered.append("alibaba/z-image-turbo")
    except ImportError:
        pass
    try:
        from ..models.krea2.provider import Krea2Provider
        from ..models.krea2.runner import DESCRIPTORS, register_krea2

        register_krea2(registry, store, policy)
        for variant, descriptor in DESCRIPTORS.items():
            requirements.register(descriptor.type, Krea2Provider(variant))
            registered.append(descriptor.type)
    except ImportError:
        pass
    try:
        from ..models.moss.provider import MossProvider
        from ..models.moss.runner import MOSS_IMAGE, register_moss

        register_moss(registry, store, policy)
        requirements.register(MOSS_IMAGE.type, MossProvider())
        registered.append(MOSS_IMAGE.type)
    except ImportError:
        pass
    try:
        from ..models.flux2.provider import Flux2Provider
        from ..models.flux2.runner import FLUX2, register_flux2

        register_flux2(registry, store, policy)
        requirements.register(FLUX2.type, Flux2Provider())
        registered.append(FLUX2.type)
    except ImportError:
        pass
    try:
        from ..models.flux1.provider import Flux1Provider
        from ..models.flux1.runner import FLUX1, register_flux1

        register_flux1(registry, store, policy)
        requirements.register(FLUX1.type, Flux1Provider())
        registered.append(FLUX1.type)
    except ImportError:
        pass
    try:
        from ..models.minimaxh3.provider import MiniMaxH3Provider
        from ..models.minimaxh3.runner import VARIANTS, register_minimax_h3

        register_minimax_h3(registry, store, policy)
        for variant in VARIANTS:
            requirements.register(variant.node_type, MiniMaxH3Provider(variant.partition))
            registered.append(variant.node_type)
    except ImportError:
        pass
    try:
        from ..models.ltx25.provider import Ltx25Provider
        from ..models.ltx25.runner import VARIANTS as LTX25_VARIANTS
        from ..models.ltx25.runner import register_ltx25

        register_ltx25(registry, store, policy)
        for variant in LTX25_VARIANTS:
            requirements.register(variant.node_type, Ltx25Provider())
            registered.append(variant.node_type)
    except ImportError:
        pass
    try:
        from ..models.zimage.primitives import register_zimage_primitives

        register_zimage_primitives(registry, store, policy)
        registered.append("primitives:z-image")
    except ImportError:
        pass
    try:
        from ..models.preprocess.requirements import PreprocessProvider
        from ..models.preprocess.runner import register_control_apply

        register_control_apply(registry, store, policy)
        # Torch-free: the annotator downloads are offered even when the runtime extra is absent.
        requirements.register("control/apply", PreprocessProvider())
        registered.append("control/apply")
    except ImportError:
        pass
    return registered


def _load_extensions(
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    requirements: RequirementsRegistry,
    rpc: Any,
    events: Any,
) -> list[LoadedExtension]:
    try:
        from ..extensions.loader import load_extensions
    except ImportError:
        return []
    return load_extensions(
        registry,
        store,
        policy,
        requirements=requirements,
        rpc=rpc,
        events=events,
    )
