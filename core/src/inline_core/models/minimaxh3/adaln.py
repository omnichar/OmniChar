"""Shrink MiniMax H3's AdaLN branch, which is 40 percent of the checkpoint.

Every block carries an ``adaln_proj.linear`` of ``[96768, 2688]``: 520 MB in bf16, 26 GB across the
50 blocks, against 40 GB for everything else. It exists to turn one timestep embedding into six
modulation vectors, and it is applied to ``silu(temb)`` and nothing else.

``silu(temb)`` turns out to live in a **five dimensional** subspace. Its singular values over the
whole timestep range are ``[23.0, 2.7, 1.7, 0.5, 0.4, 0, 0, ...]``, so a rank-8 basis captures it
exactly. Projecting through that basis first lets each block store ``[96768, 8]`` instead, which is
1.5 MB rather than 520 MB, and takes the transformer from 66.3 GB to about 40 GB.

MiniMax ship a pruned build that does the same thing with a 1025-row lookup table. This derives the
factorisation from the bf16 weights instead, which means:

* no dependency on their pruned or ``convrot`` builds being present,
* **no discrete timestep grid**. They index a table, so a sampler whose sigmas fall between rows
  needs snapping or interpolation. Projecting a continuous ``t`` through the basis is exact for any
  timestep, so the sampler is unconstrained.

Their tables are **not** compared against; nothing here reads their ``convrot`` build. What is
measured is this factorisation against the unfactorised weights: it perturbs the modulation by
1.095e-4 relative, where one bf16 ulp of re-rounding moves it 1.530e-3, so the change sits a factor
of fourteen below the ambiguity the checkpoint's own storage already carries.
``scripts/minimax_h3_adaln_gate.py`` computes both and renders the same seed each way; its numbers
land in ``outputs/minimax-h3-bench/adaln-gate/``.

Note the pixel measure disagrees in direction and is not settled: the rendered clips differ by
0.0385 mean absolute against 0.0258 for that same one-ulp perturbation. Low-rank truncation is
systematic where rounding is not, so it can compound across the 50 blocks and every step in a way
the modulation figure does not capture. The gate ran at 8 steps against production's 20.

The vendored port is **not** edited for this. The factorised module is swapped in after the model is
built, so ``vendor/`` stays verbatim.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .vendor.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
)

logger = logging.getLogger("inline_core.minimaxh3")

#: Rank kept. Five directions carry the energy; eight matches the published build and leaves slack.
RANK = 8

#: Points used to sample the timestep range when deriving the basis. The map is smooth and rank-5,
#: so this only has to be dense enough to span it; the result is then exact for continuous ``t``.
SAMPLES = 1025


@torch.no_grad()
def decompose(
    time_proj: nn.Module, time_embedder: nn.Module, *, samples: int = SAMPLES
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SVD of ``silu(temb)`` over the timestep range: singular values, and right vectors.

    Split out from ``derive_basis`` so the rank claim in this module's docstring is a number anyone
    can print from the real weights rather than a sentence to take on trust.
    """
    # A parameterless module is legitimate (a stub, an Identity), so placement falls back rather
    # than raising on an empty parameter list.
    reference = next(time_embedder.parameters(), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = reference.dtype if reference is not None else torch.float32
    grid = torch.linspace(0, 1, samples, device=device)
    embedded = time_embedder(time_proj(grid).to(dtype))
    activated = torch.nn.functional.silu(embedded).float()
    _, singular, right = torch.linalg.svd(activated, full_matrices=False)
    return singular, right


@torch.no_grad()
def derive_basis(
    time_proj: nn.Module, time_embedder: nn.Module, *, rank: int = RANK, samples: int = SAMPLES
) -> torch.Tensor:
    """An orthonormal basis for ``silu(temb)``, as ``[time_embed_dim, rank]``.

    Sampled over ``t`` in [0, 1], which is the range H3's schedulers step across.
    """
    singular, right = decompose(time_proj, time_embedder, samples=samples)
    kept = float((singular[:rank] ** 2).sum() / (singular**2).sum())
    if kept < 0.999:
        raise ValueError(
            f"A rank-{rank} basis captures only {kept:.4f} of this timestep map, so "
            "factorising the AdaLN branch would change the model. Refusing rather than "
            "degrading it silently."
        )
    logger.info("AdaLN basis: rank %d captures %.6f of the timestep map", rank, kept)
    return right[:rank].T.contiguous()


class FactorisedAdaLN(nn.Module):
    """``adaln_proj`` with its projection taken through a low-rank basis.

    Same outputs as the module it replaces, in the same order, so nothing downstream changes.
    """

    def __init__(self, hidden_size: int, out_features: int, basis: torch.Tensor) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("basis", basis, persistent=True)
        self.linear = nn.Linear(basis.shape[1], out_features, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        activated = torch.nn.functional.silu(temb)
        projected = (activated.to(self.basis.dtype) @ self.basis).to(self.linear.weight.dtype)
        out = self.linear(projected).view(-1, 6 * self.hidden_size)
        return out.chunk(6, dim=-1)


@torch.no_grad()
def factorise_block(block: Any, basis: torch.Tensor) -> FactorisedAdaLN:
    """Replace one block's AdaLN projection with its factorised equivalent."""
    original = block.adaln_proj
    weight, bias = original.linear.weight, original.linear.bias
    # The basis is derived once for the whole model; under split residency the blocks it is applied
    # to are not all on the same device as the module it came from.
    basis = basis.to(device=weight.device)
    replacement = FactorisedAdaLN(original.hidden_size, weight.shape[0], basis.to(weight.dtype))
    # W @ V: the same map, expressed in the basis the input is projected into.
    replacement.linear.weight.data = (weight.float() @ basis.float()).to(weight.dtype)
    replacement.linear.bias.data = bias.detach().clone()
    return replacement.to(device=weight.device)


@torch.no_grad()
def factorise(model: Any, *, rank: int = RANK) -> int:
    """Factorise every transformer block's AdaLN branch. Returns the bytes saved."""
    basis = derive_basis(model.time_proj, model.time_embedder, rank=rank)
    saved = 0
    def nbytes(module: Any) -> int:
        weight = module.adaln_proj.linear.weight
        return weight.numel() * weight.element_size()

    for block in model.transformer_blocks:
        before = nbytes(block)
        block.adaln_proj = factorise_block(block, basis)
        saved += before - nbytes(block)
    logger.info(
        "AdaLN factorised: %d blocks, %.1f GB saved",
        len(model.transformer_blocks), saved / 1e9,
    )
    return saved


# --- the published pruned builds ------------------------------------------------------------
#
# MiniMax ship `pruned` checkpoints that do this same rank-8 reduction ahead of time, and go one
# step further: the timestep path itself is gone. There is no `time_embedder` in the file at all,
# only `adaln_t_table [1025, 8]`, holding `silu(temb)` already projected into their basis at 1025
# points across t in [0, 1]. So the branch cannot be rebuilt as a basis applied to a `silu(temb)` we
# compute; the table has to be read directly.
#
# Off-grid timesteps are interpolated. Measured against the full bf16 weights at 24 random t, the
# table reaches 1.636e-4 relative with linear interpolation and 1.874e-4 taking the nearest row,
# where one bf16 ulp of the reference is 1.307e-3. The grid is dense enough that interpolating costs
# nothing and removes the sampler constraint a lookup would otherwise impose.

#: Rows in the published table. Checked, not assumed: a build on a different grid must not be read
#: as though it were on this one.
TABLE_ROWS = 1025


class TableEmbedder(nn.Module):
    """Stands in for ``time_proj`` + ``time_embedder``, returning the pruned build's rank-8 row.

    ``linear_1`` exists because the port reads ``time_embedder.linear_1.weight.dtype`` to cast its
    input; it carries the table's dtype and nothing else, which keeps ``vendor/`` verbatim.
    """

    #: Declared so the buffer reads as a tensor; ``register_buffer`` alone types as ``Module``.
    table: torch.Tensor

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=True)
        self.linear_1 = nn.Linear(1, 1, bias=False, dtype=table.dtype)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        rows = self.table.shape[0]
        position = timestep.to(self.table.dtype).flatten() * (rows - 1)
        low = position.floor().long().clamp(0, rows - 2)
        frac = (position - low).unsqueeze(1)
        return self.table[low] * (1 - frac) + self.table[low + 1] * frac


class TabulatedModulation(MiniMaxH3AdaLayerNormModulation):
    """``adaln_proj`` reading a table row. It already holds ``silu(temb)``, so no activation."""

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.linear(temb.to(self.linear.weight.dtype)).view(-1, 6 * self.hidden_size)
        return out.chunk(6, dim=-1)


class TabulatedNormOut(MiniMaxH3AdaLayerNormOut):
    """``norm_out`` reading a table row, otherwise the port's own forward."""

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


@torch.no_grad()
def tabulate(model: Any, table: torch.Tensor) -> None:
    """Rebuild the timestep path around a pruned build's table, on the meta device before streaming.

    Every module the table feeds changes shape, so this has to happen before any weight is placed:
    the rank-8 ``adaln_proj`` in the file would otherwise be assigned into a ``[96768, 2688]`` slot.
    """
    if table.ndim != 2 or table.shape[0] != TABLE_ROWS:
        raise ValueError(
            f"adaln_t_table is {tuple(table.shape)}, not [{TABLE_ROWS}, rank]. This build is on a "
            "different timestep grid from the one measured, and reading it as if it were not would "
            "shift the modulation at every step while still rendering."
        )
    rank = int(table.shape[1])
    model.time_proj = nn.Identity()
    model.time_embedder = TableEmbedder(table)
    for block in model.transformer_blocks:
        block.adaln_proj = _retyped(block.adaln_proj, TabulatedModulation, rank)
    model.norm_out = _retyped(model.norm_out, TabulatedNormOut, rank)


def _retyped(module: Any, cls: type, rank: int) -> Any:
    """The same module with its projection narrowed to ``rank`` inputs, still on meta."""
    replacement = module
    replacement.__class__ = cls
    old = module.linear
    with torch.device("meta"):
        replacement.linear = nn.Linear(rank, old.out_features, bias=old.bias is not None)
    return replacement


#: Parameters ``tabulate`` creates that no checkpoint fills.
TABULATED_SELF_COMPUTED = ("time_embedder.linear_1.weight", "time_embedder.table")
