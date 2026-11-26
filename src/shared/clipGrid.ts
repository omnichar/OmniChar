/**
 * The clip frame grids, mirrored from Core so the Trainer can show what a clip length actually
 * resolves to before a run starts.
 *
 * A video VAE only encodes `grid * n + offset` frames, and Core snaps **down** onto that grid.
 * Asking H3 for 5s therefore trains on 4.458s, and the shortfall comes off the end. That was silent
 * until the number appeared next to the field, and a user lost the action in the last half second of
 * their clips to it.
 *
 * Note the direction. Generation snaps **up** and then clamps (`models/video_params.py`), because a
 * request should be honoured where it is legal. Training snaps down, because a clip does not have
 * frames the file never held. The two are separate on purpose.
 */

import type { TrainingArch } from './types'

/** One architecture's frame grid, mirroring Core's `training/arch.py::ClipGrid`. */
export interface ClipGrid {
  fps: number
  grid: number
  offset: number
}

/**
 * Every architecture's grid, or null for one that trains on stills alone. Exhaustive over
 * `TrainingArch` so adding an architecture without deciding this is a type error, not a silent
 * `undefined` that would make the Trainer promise a clip length nothing snaps.
 */
export const CLIP_GRIDS: Record<TrainingArch, ClipGrid | null> = {
  'z-image': null,
  krea2: null,
  flux1: null,
  flux2: null,
  'minimax-h3': { fps: 24, grid: 17, offset: 5 },
  'ltx-2-5': { fps: 24, grid: 8, offset: 1 },
  'moss-image': null,
}

/** The shortest clip a grid encodes: one whole chunk plus the head. */
export function minClipFrames(grid: ClipGrid): number {
  return grid.grid + grid.offset
}

/** The largest `grid * n + offset` that fits `frames`, or the floor when nothing does. */
export function snapClipFramesFor(grid: ClipGrid, frames: number): number {
  const floor = minClipFrames(grid)
  if (!Number.isFinite(frames) || frames < floor) return floor
  return Math.floor((frames - grid.offset) / grid.grid) * grid.grid + grid.offset
}

/** What `seconds` of clip actually trains on, as `{frames, seconds}` on `grid`. */
export function resolveClipLengthFor(
  grid: ClipGrid,
  seconds: number,
): { frames: number; seconds: number } {
  const frames = snapClipFramesFor(grid, Math.round(seconds * grid.fps))
  return { frames, seconds: frames / grid.fps }
}

/** The grid for an architecture, or null when it trains on stills. */
export function clipGridFor(arch: TrainingArch | undefined): ClipGrid | null {
  return arch ? CLIP_GRIDS[arch] : null
}

// --- MiniMax H3, kept as named exports ------------------------------------------------------------
// H3 was the only clip architecture before LTX-2.5, so these are what the Trainer and its test were
// written against. They stay as the H3 grid applied to the generic helpers above.

/** H3's fixed frame rate. Never a parameter: it is what the grid is defined against. */
export const H3_FPS = 24

/** The shortest clip H3's video VAE encodes: the first `17n + 5` above a single frame. */
export const H3_MIN_CLIP_FRAMES = 22

/** The largest `17n + 5` that fits `frames`, or `H3_MIN_CLIP_FRAMES` when nothing does. */
export function snapClipFrames(frames: number): number {
  return snapClipFramesFor(CLIP_GRIDS['minimax-h3'] as ClipGrid, frames)
}

/** What `seconds` of clip actually trains on, as `{frames, seconds}` on the grid. */
export function resolveClipLength(seconds: number): { frames: number; seconds: number } {
  return resolveClipLengthFor(CLIP_GRIDS['minimax-h3'] as ClipGrid, seconds)
}
