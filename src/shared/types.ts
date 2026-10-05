/**
 * Domain types shared by the main and renderer processes.
 *
 * The mental model (see CLAUDE.md):
 *   Project → Sequence → Frame → Take[]
 *   A Frame is a *slot with a history of takes*, never a single file.
 *   The timeline points at a Frame's `heroTakeId` (the current chosen take).
 */

export type FrameKind = 'image' | 'video' | 'audio'
export type AssetKind = 'image' | 'video' | 'audio'

/** A project is a portable `.inlinestudio` folder; this is its DB-backed metadata. */
export interface Project {
  id: string
  name: string
  /** Absolute path to the `.inlinestudio` folder on disk. */
  path: string
  createdAt: number
  updatedAt: number
  /** True once the Workflows popup has auto-opened here, so a dismissal sticks. */
  workflowsPrompted?: boolean
}

/** Lightweight entry for the "recent projects" list (stored in app userData). */
export interface RecentProject {
  name: string
  path: string
  lastOpenedAt: number
}

export interface Sequence {
  id: string
  projectId: string
  name: string
  /** Order within the project. */
  position: number
}

export interface Frame {
  id: string
  sequenceId: string
  name: string
  kind: FrameKind
  /** Order within the sequence. */
  position: number
  /** The imported source asset this frame edits (its Input row), if any. */
  inputAssetId: string | null
  /** Currently chosen take placed on the timeline, if any. */
  heroTakeId: string | null
  /**
   * Which generation engine backs this frame. `fal` = a declarative fal.ai node (see
   * `modelId`/`params`). `core` = an Inline Core node. `unset` = a legacy frame with no engine
   * (renders as a plain viewer). (Legacy projects may still carry a `'comfy'` string from before
   * ComfyUI was removed; it renders as a plain viewer too.)
   */
  provider: 'fal' | 'core' | 'unset'
  /** For `provider:'fal'`: the registry model id (e.g. `openai/gpt-image-2`). */
  modelId: string | null
  /** For `provider:'fal'`: the node's editable param values (keyed by the NodeDef param keys). */
  params: Record<string, unknown>
  /** Workflow template this frame generates with, if chosen. */
  workflowTemplateId: string | null
  createdAt: number
  updatedAt: number
}

/**
 * One of a frame's inputs (a frame can have several). An input is either a library
 * asset (`assetId`) or a live link to another frame's output (`sourceFrameId`) - the
 * latter resolves to that frame's hero take at generate time (the refine/flow link).
 */
export interface FrameInput {
  id: string
  frameId: string
  assetId: string | null
  sourceFrameId: string | null
  position: number
  /** The model input port this was wired to, or null when untagged (drag-drop / pre-v17 rows). */
  handle: string | null
}

/** Every ComfyUI render of a frame becomes an immutable Take. */
/**
 * A generation Core still has in flight. Shaped like the progress event so the renderer can feed
 * it through the same reducer when rebuilding its queue after a page refresh.
 */
export interface ActiveGeneration {
  frameId: string
  /** 0..1, or null when the run has not reported yet (a long model load reports nothing). */
  fraction: number | null
  status?: string
}

export interface Take {
  id: string
  frameId: string
  /** Relative path under the project's `takes/` folder. */
  filePath: string
  kind: AssetKind
  /** Snapshot of the params used to generate this take (JSON). */
  params: Record<string, unknown>
  /** ComfyUI prompt id that produced this take, if generated. */
  comfyPromptId: string | null
  createdAt: number
}

/** A logical folder in the asset library (a tree; physical files stay flat on disk). */
export interface AssetFolder {
  id: string
  projectId: string
  name: string
  /** Parent folder id, or null for a root-level folder. */
  parentId: string | null
  createdAt: number
}

/** Imported media in the project library. */
export interface Asset {
  id: string
  projectId: string
  /** Logical folder this asset lives in, or null for the library root. */
  folderId: string | null
  name: string
  /** Relative path under the project's `assets/` folder. */
  filePath: string
  kind: AssetKind
  /** Poster image (first frame) for videos, so they always render. Relative path. */
  thumbPath: string | null
  /** A Chromium-playable transcode for videos in codecs the UI can't decode. Relative. */
  previewPath: string | null
  createdAt: number
}

/**
 * An item on the unified canvas. Beyond ideation items (asset/text), the canvas
 * also hosts the production graph: `frame` nodes (input/output handles), `layer`
 * group containers, and `preview` nodes that display a connected frame's output.
 */
export type MoodboardItemType =
  | 'asset'
  | 'text'
  | 'frame'
  | 'layer'
  | 'preview'
  | 'director'
  | 'trim'
  /** A text-prompt node whose output feeds a Generate node's prompt input. */
  | 'prompt'
  /** A low-level Inline Core graph node (load/sample/encode/vae); ephemeral, not a Frame. */
  | 'core'
  /** A "Load Assets" node: holds library asset refs in `data.assetIds`, feeds its hero downstream. */
  | 'loader'
  /** A "Control Space" node: a 3D pose editor whose rendered OpenPose map (`data.controlAssetId`)
   * feeds a gen node's control input. */
  | 'controlSpace'
  /** Picks a training dataset and feeds it downstream. */
  | 'train/dataset'
  /** Auto-captions a dataset's images. */
  | 'train/caption'
  /** Runs a LoRA training job (run/stop/resume, live steps + logs). */
  | 'train/lora'
  /** Plots a training run's loss curve. */
  | 'train/loss'
  | 'log/tail'
  /** Utility: read-only host telemetry (CPU/RAM/VRAM). No handles. */
  | 'resource'

/** Output settings for a video-director node (stored in its moodboard item data). */
export interface DirectorItemData {
  /** Composition width in px. */
  width: number
  /** Composition height in px. */
  height: number
  /** Frames per second. */
  fps: number
}

/**
 * One derived clip on a director node's timeline. The video layer and L1 audio are
 * derived from the videos wired into the node; L2 is the user's wired audio. Positions
 * are sequential (seconds); nothing here is persisted - it's recomputed from the wired
 * connections each time.
 */
export interface DirectorClip {
  /** Stable key for React (the source frame/asset id). */
  key: string
  /** The connector feeding this input (for per-input volume edits). */
  connectorId: string
  /** Per-input audio volume 0..1 (L1 = the video's extracted audio). */
  volume: number
  /** Source frame id (for the "Frame X" tag + canvas navigation); null for asset clips. */
  frameId: string | null
  /** Display label (frame name / asset name). */
  label: string
  kind: AssetKind
  /** Position on the layer, in seconds. */
  startTime: number
  /** Clip length, in seconds. */
  duration: number
  /** Project-relative waveform peaks JSON for this clip's audio, if any. */
  audioPeaks: string | null
  /** Fraction [start, end] of the source peaks this clip spans (a trim window; full = 0..1). */
  peaksStart: number
  peaksEnd: number
  /** Project-relative thumbnail: a filmstrip PNG for video, the still image for image clips. */
  thumbnail: string | null
}

/** The resolved source media behind a trim ("Edit Video/Audio") node, for its UI. */
export interface TrimResolved {
  /** Source frame/asset id. */
  key: string
  kind: AssetKind
  label: string
  /** Full source duration in seconds (0 if unknown). */
  durationSec: number
  /** Project-relative media path (for an in-node preview element). */
  mediaPath: string
  /** Project-relative filmstrip PNG (video), or null. */
  thumbnail: string | null
  /** Project-relative waveform peaks JSON (audio), or null. */
  audioPeaks: string | null
}

/** The derived, display-ready timeline for a director node (recomputed from connections). */
export interface DirectorTimeline {
  /** Video layer clips (in slot order). */
  video: DirectorClip[]
  /** Audio layer 2 clips (the user's wired audio). */
  l2: DirectorClip[]
  /** Volume of the extracted video audio (L1), 0..1. */
  l1Volume: number
  /** Volume of the user audio (L2), 0..1. */
  l2Volume: number
}

export interface TextItemData {
  text: string
  fontSize: number
  bold: boolean
  italic: boolean
  underline: boolean
  color: string
  align: 'left' | 'center' | 'right'
  /** Optional http(s) URL - renders the text as a clickable link. */
  link?: string
}

/** Type-specific payload for a moodboard item (text styling, layer/director settings). */
export interface MoodboardItemData {
  text?: TextItemData
  /** Display name for a layer group or director node. */
  name?: string
  /** Accent color (hex) for a layer group. */
  color?: string
  /** Output settings for a director node. */
  director?: DirectorItemData
  /** Project-relative path of a director node's last-built proxy preview MP4. */
  directorPreview?: string
  /** Project-relative path of a director node's last full-res Export MP4 (shown in a wired preview). */
  directorExport?: string
  /** Director node: volume of the extracted video audio layer (L1), 0..1 (default 1). */
  l1Volume?: number
  /** Director node: volume of the user audio layer (L2), 0..1 (default 1). */
  l2Volume?: number
  /** Trim ("Edit Video/Audio") node: in/out window in seconds (outPoint <= inPoint = "to end"). */
  trim?: { inPoint: number; outPoint: number }
  /** Prompt node: the text it feeds into a connected Generate node's prompt input. */
  promptText?: string
  /** `loader` ("Load Assets") node: ordered library asset ids it holds; the first (hero) feeds
   * downstream. A pure viewer with no generation. (Legacy: also flags an old frame-based loader.) */
  loader?: boolean
  /** `loader` node: the library asset ids it holds, in order (hero = first). */
  assetIds?: string[]
  /** `controlSpace` node: the library asset id of the last rendered OpenPose control map, fed
   * downstream into a gen node's control input. */
  controlAssetId?: string
  /** `controlSpace` node: the serialized 3D scene so the editor reopens on the saved pose.
   * `characters` is the current multi-character form; `joints` is the legacy single-character field,
   * still read for scenes saved before multi-character support. */
  controlScene?: {
    joints?: [number, number, number][]
    characters?: { joints: [number, number, number][] }[]
    fov?: number
    /** Camera pose so a saved render reproduces the exact framing on reopen. */
    camera?: { position: [number, number, number]; target: [number, number, number] }
    /** Which control map the node last rendered: an OpenPose skeleton or a depth map. */
    output?: 'pose' | 'depth'
    /** Output aspect (w/h) of the rendered map; must match the gen node's resolution or the pose
     * comes out stretched. Defaults to 1 (square). */
    aspect?: number
    /** How each character was turned relative to the camera when the map was rendered. */
    facing?: ('front' | 'three-quarter' | 'profile' | 'back' | null)[]
    /** Prompt text the facing needs, resolved at render time and appended to a downstream gen node's
     * prompt / negative prompt. A control map cannot express "facing away" - only the text encoder
     * can - so Control Space states it here. Null (or absent) = nothing to add. */
    promptHint?: { positive: string; negative: string } | null
    /** Whether the facing prompt hint is applied downstream (default true). */
    applyPromptHint?: boolean
  }
  /** Training nodes: the training dataset they operate on. */
  datasetId?: string | null
  /** `caption` node: re-caption images that already have a caption. */
  overwrite?: boolean
  /** `caption` node: which captioner to run. Read by `graph_build`, so it belongs on the wire. */
  captioner?: string
  /** Training nodes: the training run they're bound to. Persisted so the node
   * rebinds after a reload and can still offer Resume on an interrupted run. */
  runId?: string | null
  /** `trainer` node: its hyperparameters (edited in the Adjust sidebar, off the node face). */
  hyperparams?: Partial<TrainingHyperparams>
  /** Core graph node: the Inline Core node type + its param values (see coreNodes.ts). */
  core?: {
    type: string
    params: Record<string, unknown>
    /** The active media this node produced - shown large and flowed downstream. */
    output?: CoreTakeRef
    /** Recent renders (newest first); `output` points at the active one. Drives the take-history
     * strip on generation nodes. */
    outputs?: CoreTakeRef[]
    /** The last submitted run, which has no take yet. Drives the strip's Current slot. */
    pending?: CorePendingRun
  }
}

/** A run as submitted, held until it lands as a take.
 *
 * Recorded because a running generation's settings live nowhere else: the node's params can be
 * edited while it renders, and until it finishes there is no take carrying the recipe it used. */
export interface CorePendingRun {
  /** Node params at submit. */
  params?: Record<string, unknown>
  /** Upstream prompt text at submit. */
  prompt?: string
  startedAt: number
  status: 'draft' | 'running' | 'cancelled' | 'failed'
}

/** One render in a Core node's history: the media file plus the recipe fields that let switching to
 * it restore the node's settings and its prompt (the reproducible-recipe feature). */
export interface CoreTakeRef extends TakeContinuity {
  takeId: string
  filePath: string
  kind: 'image' | 'video' | 'audio'
  /** Stamped at generation (ms); absent on renders made before it was tracked. */
  createdAt?: number
  /** The runner's resolved settings. Not the node's: different keys, and `model` is a runner id. */
  params?: Record<string, unknown>
  /** The node's own params at render time, which is the only sound baseline for "has this changed". */
  nodeParams?: Record<string, unknown>
  /** The prompt this take used (from the upstream prompt node), for display + restore. */
  prompt?: string
}

/** Spread onto a Core take ref; a fal take carries it inside `Take.params.continuity`. */
export interface TakeContinuity {
  /** The character file applied to this take, when one was. */
  characterId?: string
  /** 0-100 identity match against that character's references. Absent means "not measured", which
   * is a different fact from a low score, so it must never be defaulted to zero. On a video this
   * is the median across the frames that measured. */
  continuityScore?: number
  /** The score is the face term alone: the references cannot speak to this take's framing. */
  continuityFaceOnly?: boolean
  /** No face was found, so DINOv2 answered alone. That measures framing, not identity, and the
   * number must never be read as an identity match. */
  continuitySubjectOnly?: boolean
  /** 0-100 match against the character's wardrobe references, set only when the clothes were in
   * frame. Reported beside the score, never inside it: the score is face 0.8 + subject 0.2 and
   * every number measured so far would move if a third term were folded in. */
  continuityWardrobe?: number
  /** Video: how many sampled frames carried a measurable face. */
  continuityFrames?: number
  /** Video: sampled frames with no face at all - turned away, blurred, or out of frame. Counted
   * and reported, never scored as zero, and never part of the statistics below. */
  continuityNoFace?: number
  /** Video: the mean across measured frames. What "held at" means. */
  continuityMean?: number
  /** Video: the worst measured frame. The real signal - a mean hides a visible identity break. */
  continuityMin?: number
  /** Video: when that worst frame occurred, in seconds. */
  continuityMinAt?: number
  /** Which gallery the identity term was measured against. */
  continuityGallery?: 'frozen' | 'refs'
}

/** One saved character in `models/characters/`, as the library panel lists it. */
export interface CharacterSummary {
  /** The filename, which is also the value a node's Character dropdown stores. */
  file: string
  charId?: string
  name: string
  /** The references the user curated. Harvested takes are counted separately on purpose. */
  refs: number
  /** Approved takes taken into the pool. Absent or 0 on a character that never harvested one. */
  harvested?: number
  /** Reference positions whose face agrees least with the rest, by `character/verify-refs`. */
  flagged?: number[]
  createdAt?: number
  modifiedAt?: number
  description?: string
  /** Non-blocking nudges toward a stronger character ("Add a profile view"). */
  hints?: string[]
  sizeBytes?: number
  /** The references moved on from what scoring and the payload were built from. */
  needsRebuild?: boolean
  /** Seconds of the stored voice payload; null or absent when the character has no voice. */
  voiceSeconds?: number | null
  /** Set when the file could not be read. The row is still listed, so a corrupt file is visible. */
  error?: string
}

export interface MoodboardItem {
  id: string
  projectId: string
  /** Which canvas this item belongs to (defaults to the Studio moodboard). */
  surface: string
  type: MoodboardItemType
  /** Set when type === 'asset'. */
  assetId: string | null
  /** Set when type === 'frame' (and reused by 'preview' resolution). */
  frameId: string | null
  /** Containing layer item id, if this item lives inside a layer group. */
  parentId: string | null
  data: MoodboardItemData
  x: number
  y: number
  width: number
  height: number
  rotation: number
  zIndex: number
  createdAt: number
  updatedAt: number
}

/** An arrow/line connecting two moodboard items. */
export interface MoodboardConnector {
  id: string
  projectId: string
  /** Derived from the nodes it joins - a connector never spans two canvases. */
  surface: string
  fromItemId: string
  toItemId: string
  label: string | null
  data: Record<string, unknown>
  createdAt: number
}

/** The full board state for a project. */
export interface MoodboardSnapshot {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
}

/** Main → renderer: director timeline render progress (0..1), correlated by node id. */
export interface TimelineProgressEvent {
  ownerItemId: string
  fraction: number
}

/** Main → renderer: per-node generation progress (0..1), correlated by the running frame's id. */
export interface GenerationProgressEvent {
  /** The node the run was started from - what the run is filed under. */
  frameId: string
  fraction: number
  /** Optional short status label from the fal queue (e.g. "in queue", "generating"). */
  status?: string
  /** The node executing right now, which is only `frameId` for a single-node graph. */
  nodeId?: string
}

/** Main → renderer: a node in the run finished and produced a take. */
export interface GenerationNodeDoneEvent {
  frameId: string
  takeId: string
  /** The node that finished, so the canvas can stop showing that one as running. */
  nodeId?: string
}

/** Main → renderer: the whole run (target frame + its upstream chain) completed. */
export interface GenerationDoneEvent {
  targetFrameId: string
}

/** Main → renderer: the run failed; `frameId` is the node that failed, if any. */
export interface GenerationErrorEvent {
  targetFrameId: string
  frameId?: string
  error: string
  runId?: string
}

/** Main → renderer: a run was cancelled, so its node can stop showing as busy. */
export interface GenerationCancelledEvent {
  targetFrameId: string
  runId?: string
}

/** What a run is doing. Mirrors Core's `RunStatus`, plus `interrupted` for a run lost to a crash. */
export type ActivityStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled' | 'interrupted'

/**
 * One run in the activity panel. Covers Core generations, fal generations and training runs, from
 * any project - Core is a single process, so its queue spans whichever project happens to be open.
 */
export interface ActivityRun {
  runId: string
  kind: 'generation' | 'training'
  engine: 'core' | 'fal'
  /** `api` is a run submitted straight to `POST /v1/runs` rather than started from this UI. */
  origin: 'studio' | 'api'
  status: ActivityStatus
  title: string
  /** 0..1, or null when the run has not reported yet (a long model load reports nothing). */
  fraction: number | null
  statusLabel: string | null
  /** 0-based place in the pending line; null once the run has left the queue. */
  queuePosition: number | null
  queuedAt: number
  startedAt: number | null
  endedAt: number | null
  error: string | null
  takeId: string | null
  /** Studio-origin only. An API run has no project and no canvas node behind it. */
  projectId: string | null
  projectName: string | null
  projectPath: string | null
  itemId: string | null
  surface: string | null
}

/** Main → renderer: the live run list changed. Carries the whole list, so it replaces state. */
/** One phase of a character encode, streamed while the create/edit RPC is still in flight. */
/** What one architecture can apply for a character, and whether its adapter is worth trusting. */
/** One model in the published registry: where it comes from, and whether it is already here. */
export interface RegistryModel {
  id: string
  label: string
  filename: string
  /** The `models/` subfolder it lands in. */
  category: string
  kind: 'hf_file' | 'hf_folder' | 'url'
  repo: string
  path: string
  url: string
  verified: boolean
  optional: boolean
  sizeBytes: number | null
  updated: string
  /** Entries sharing a group are the same model at a different precision. */
  group: string
  precision: string
  present?: boolean
}

/** A weight file something asked for that is not on disk. */
export interface MissingModel {
  wanted: string
  /** Where it belongs, relative to the models root. Shown even when nothing can be downloaded. */
  path: string
  matches: { model: RegistryModel; exact: boolean; present: boolean }[]
}

export interface CharacterProgressEvent {
  /** The character's name, which is all the client has to match on before a file exists. */
  name: string
  /** 0 to 1. */
  fraction: number
  status: string
}

export interface ActivityChangedEvent {
  runs: ActivityRun[]
}

/** One file under a models root. */
export interface ModelTreeFile {
  name: string
  path: string
  kind: 'file'
  sizeBytes: number
  mtime: number
}

/** A folder under a models root. Only folders containing weights are listed. */
export interface ModelTreeDir {
  name: string
  path: string
  kind: 'dir'
  children: ModelTreeNode[]
  fileCount: number
}

export type ModelTreeNode = ModelTreeFile | ModelTreeDir

/** One scanned models root. Only the first is writable; the rest are read-only extras. */
export interface ModelTreeRoot {
  path: string
  label: string
  writable: boolean
  exists: boolean
  categories: ModelTreeDir[]
}

/**
 * A LoRA written partway through a training run, so a run can be judged before it finishes.
 * Lives in the project's working dir, which no model picker scans - `exportSnapshot` copies one
 * into `models/loras/` to make it selectable.
 */
export interface TrainingSnapshot {
  runId: string
  step: number
  /** Project-relative path, for the download link. */
  path: string
  sizeBytes: number
  createdAt: number
  /** Where it was copied under `models/`, set when it was exported automatically on creation. */
  loraPath?: string
}

/** Main → renderer: progress (0..1) of an explicit model download (the node's model popup). */
export interface ModelDownloadProgressEvent {
  nodeType: string
  componentId: string
  fraction: number
  status?: string
}

/** Main → renderer: a model component finished downloading (files are now under models/). */
export interface ModelDownloadDoneEvent {
  nodeType: string
  componentId: string
}

/** Main → renderer: a model download failed. */
export interface ModelDownloadErrorEvent {
  nodeType: string
  componentId: string
  error: string
}

// LoRA training ------------------------------------------------------------

/** Turbo needs a training adapter to avoid "turbo drift"; a de-turbo base trains without one. */
/**
 * The model family a LoRA is trained for. `minimax-h3` and `ltx-2-5` are the video models: they
 * train on stills or clips, and the adapter applies to every node of that family at generation time.
 */
export type TrainingArch = 'z-image' | 'krea2' | 'flux1' | 'flux2' | 'minimax-h3' | 'ltx-2-5'

/**
 * Which shape an LTX-2.5 run trains in. `clip` learns a look and how it moves, from single clips.
 * `control` learns a transform from paired reference and target clips - upstream calls this an
 * IC-LoRA - so every dataset item needs a second clip wired to it as the reference.
 */
export type TrainingMode = 'clip' | 'control'

/** One row a source produced, held in the dialog until the reader commits it to a dataset. */
export interface StagedDatasetItem {
  assetId: string
  name: string
  caption: string
  referenceAssetId: string | null
}

/** A Hugging Face dataset repo, summarised before anyone commits to downloading it. */
export interface DatasetRepoPreview {
  repo: string
  /** Trainable items, excluding reference clips (which are the `before` half of a pair). */
  items: number
  /** How many of those items have a reference, so a control set is recognisable up front. */
  pairs: number
  bytes: number
  metadataFile: string | null
  /** Set when the repo cannot be used, with the reason to show instead of a count. */
  problem: string | null
}

/**
 * Which base checkpoint a run trains against. `raw` is Krea 2's undistilled base (the recommended
 * path - the LoRA then applies to Turbo at generation time); the turbo modes train against a
 * distilled checkpoint and need a training adapter to avoid turbo drift. `raw_9b` is FLUX.2 klein
 * Base 9B - a size rather than a distillation choice, and `raw` stays 4B so saved runs resolve to
 * the checkpoint they were trained against.
 */
export type TrainingBaseMode = 'turbo_adapter' | 'deturbo' | 'raw' | 'raw_9b'

/**
 * Precision of the frozen base during training. The LoRA itself is always full precision, so `nf4`
 * only costs base fidelity - and it is what puts Krea 2 (a 26GB base) on a 16GB card. `auto` sizes
 * the base plus its activations against your GPU and picks for you.
 */
export type TrainingBaseQuant = 'auto' | 'none' | 'nf4'

/** Stream saved activations to host RAM so a bf16 base fits a smaller card. */
export type TrainingOffload = 'auto' | 'on' | 'off'

/**
 * Which Linears the adapter attaches to. `full` adapts the feed-forward and projection layers as
 * well, which is stronger on short style runs; `attention` is the Krea 2 authors' advice for long
 * runs, where adapting everything starts to cost prompt adherence.
 */
export type TrainingLoraScope = 'full' | 'attention'

export type TrainingStatus =
  | 'queued'
  | 'training'
  | 'done'
  | 'failed'
  | 'cancelled'
  /** Died to a crash/OOM/restart; resumable from its last checkpoint. */
  | 'interrupted'

/** A named set of images or clips (+ captions) that trains one LoRA. */
export interface TrainingDataset {
  id: string
  projectId: string
  name: string
  /** Token injected into every caption (the character/style trigger); may be empty. */
  triggerWord: string
  /**
   * `clip` trains on one media file per item. `control` is LTX-2.5's IC-LoRA: every item also
   * needs a reference clip, and the model learns the transform between the pair. Defaults to
   * `clip`, so a dataset made before this existed keeps its meaning.
   */
  mode: TrainingMode
  createdAt: number
  updatedAt: number
}

/** One image or clip in a dataset, with its caption. Backed by a library asset. */
export interface TrainingDatasetItem {
  id: string
  datasetId: string
  assetId: string
  /**
   * The `before` of a Control LoRA pair, or null on a clip-mode item. A control run refuses to
   * start while any item is unpaired, because a missing reference does not fail loudly once
   * training is under way - the pair is simply absent.
   */
  referenceAssetId: string | null
  caption: string
  position: number
  createdAt: number
}

/** A local auto-caption model the Trainer can offer. Defined in Core's `training/caption.py`. */
export interface CaptionerModel {
  id: string
  label: string
  repo: string
}

/** LoRA training hyperparameters (persisted as JSON on a run). */
export interface TrainingHyperparams {
  /** Defaults to `z-image` when absent, so runs saved before Krea 2 keep working. */
  arch?: TrainingArch
  baseMode: TrainingBaseMode
  /** Defaults to `auto`. Only Krea 2 has a 4-bit path; Z-Image always trains in bf16. */
  baseQuant?: TrainingBaseQuant
  /**
   * Defaults to `auto`. Offloads saved activations to host RAM during the step so a full-precision
   * base fits a card that could not otherwise hold base + activations. `auto` turns it on only when
   * the bf16 plan would not fit resident. Trades some speed for the VRAM.
   */
  offload?: TrainingOffload
  /** Defaults to `full`. */
  loraScope?: TrainingLoraScope
  /** 0 to 1: how often a step trains on an empty caption, so the LoRA holds without the trigger. */
  captionDropout?: number
  /** Mirror every image, doubling the dataset. Wrong for text or anything asymmetric. */
  flipAugment?: boolean
  /** Keep a usable LoRA at every checkpoint, not just at the end. Off by default: each one is a
   *  full adapter, so a long run costs real disk. */
  saveSnapshots?: boolean
  /** LoRA rank (dim). */
  rank: number
  /** LoRA alpha; defaults to `rank`. */
  alpha: number
  learningRate: number
  steps: number
  batchSize: number
  /** Square training resolution in px (e.g. 1024). */
  resolution: number
  /**
   * Seconds of each video clip to train on, for archs that accept clips. Snapped **down** to the
   * model's frame grid, and ignored entirely for stills. MiniMax H3's floor is 0.92s.
   */
  clipSeconds?: number
  /**
   * Which end of a clip the training window is taken from. Snapping down always discards some of
   * the clip, so `end` exists for footage whose action is at the finish. Defaults to `start`.
   */
  clipWindow?: 'start' | 'end'
  /**
   * Which shape an LTX-2.5 run trains in. Defaults to `clip`. It changes which Linears the adapter
   * reaches as well as what the dataset must hold, so it is a hyperparameter rather than a view.
   */
  trainingMode?: TrainingMode
  /** Checkpoint every N steps. */
  saveEvery: number
  /** GPU indices to train on; `[]` = auto (first available). */
  gpuIds: number[]
  /** Filename (no extension) for the produced LoRA. Blank = derived from the run name. */
  outputName?: string
}

/** A training run over a dataset, producing a LoRA `.safetensors`. */
export interface TrainingRun {
  id: string
  projectId: string
  datasetId: string
  name: string
  status: TrainingStatus
  hyperparams: TrainingHyperparams
  /** `models/loras/`-relative path of the produced LoRA, once done. */
  outputLoraPath: string | null
  progressFraction: number
  progressStatus: string | null
  step: number
  totalSteps: number
  error: string | null
  createdAt: number
  updatedAt: number
}

/** Main → renderer: training progress for a run (0..1) + step + latest loss (for the loss curve). */
export interface TrainingProgressEvent {
  runId: string
  fraction: number
  step: number
  totalSteps: number
  loss?: number
  status?: string
}

/** Main → renderer: a mid-training sample preview image was written (project-relative path). */
export interface TrainingSampleEvent {
  runId: string
  step: number
  path: string
}

/** Main → renderer: auto-caption progress for a dataset (one tick per captioned image). */
export interface CaptionProgressEvent {
  datasetId: string
  /** Images captioned so far, of `total` queued this run. */
  done: number
  total: number
  /** The item just captioned, if this tick is for one. */
  itemId: string | null
  /** True on the final tick, when the captioner has exited. */
  finished: boolean
}

/** Main → renderer: one stdout line from the trainer subprocess (shown in the Trainer node). */
export interface TrainingLogEvent {
  runId: string
  line: string
}

/** Main → renderer: a training run finished; `outputLoraPath` is the produced LoRA. */
export interface TrainingDoneEvent {
  runId: string
  outputLoraPath: string
}

/** Main → renderer: a graph run's Train LoRA node started a run and this node now owns it. */
export interface TrainingNodeBoundEvent {
  itemId: string
  runId: string
}

/** Main → renderer: a training run failed. */
export interface TrainingErrorEvent {
  runId: string
  error: string
}

/** Per-GPU live stats (NVML). */
export interface GpuStat {
  index: number
  name: string
  /** GPU utilization %, 0..100. */
  utilization: number
  /** VRAM used / total, in bytes. */
  memoryUsed: number
  memoryTotal: number
  /** Degrees C, or null if unavailable. */
  temperature: number | null
}

/** Main → renderer: periodic host + GPU telemetry (the Resources node). */
export interface SystemStatsEvent {
  /** CPU utilization %, 0..100. */
  cpu: number
  /** System RAM used / total, in bytes. */
  ramUsed: number
  ramTotal: number
  gpus: GpuStat[]
}

/** Main → renderer: extension install progress. Payload shapes live in `extensions.ts`. */
export type {
  InstallProgressEvent as ExtensionInstallProgressEvent,
  InstallSuccess as ExtensionInstallDoneEvent,
  InstallFailure as ExtensionInstallErrorEvent,
} from './extensions'

export interface WorkflowTemplate {
  id: string
  projectId: string
  name: string
  /** Raw ComfyUI workflow JSON. */
  graph: Record<string, unknown>
  /** Param schema marking which node inputs are user-editable. */
  params: WorkflowParam[]
}

export interface WorkflowParam {
  key: string
  label: string
  type: 'text' | 'number' | 'seed' | 'image' | 'model'
  /** Path into the graph JSON this param drives, e.g. "6.inputs.text". */
  nodePath: string
  default?: string | number
}

/** App-global settings (stored in Electron userData, not per-project). */
export interface AppSettings {
  /** The Inline Core (/v1) engine URL. */
  coreUrl: string
}

/** Whether an API key (e.g. fal.ai) is saved, and how it's stored. */
export interface ApiKeyStatus {
  /** A key is saved. */
  configured: boolean
  /** False when the OS lacks a secure keystore and the key is stored plaintext. */
  encrypted: boolean
}

/** Auto-update events (main → renderer). */
export interface UpdateAvailableEvent {
  version: string
  /** macOS (unsigned) can't self-install, so the renderer opens the releases page instead. */
  notifyOnly: boolean
}
export interface UpdateProgressEvent {
  /** 0–100. */
  percent: number
  transferred: number
  total: number
}
export interface UpdateDownloadedEvent {
  version: string
}

/** Result of pinging the configured Inline Core engine. */
export interface CoreStatus {
  running: boolean
  url: string
}

/** Absolute media directories of the open project. */
export interface ProjectMediaDirs {
  /** Where Inline Studio keeps imported inputs - point ComfyUI's --input-directory here. */
  inputDir: string
  /** Where Inline Studio keeps generated outputs - point ComfyUI's --output-directory here. */
  outputDir: string
}

/** Result of exporting a whole project to a portable .zip. */
export interface ProjectExportResult {
  /** Absolute path of the written .zip archive. */
  path: string
}

/** Summary of an "export frames to folder" run. */
export interface ExportResult {
  /** Absolute directory the files were written to. */
  dir: string
  /** Count of frame outputs exported. */
  exported: number
  /** Names of frames skipped because they had no Output yet. */
  skipped: string[]
}

// --- published workflow catalogue (omnichar.org) ------------------------------------------

/** Which axis a category sits on. The Workflows rail draws the two as separate blocks. */
export type WorkflowCategoryKind = 'type' | 'model'

export interface WorkflowCategory {
  slug: string
  name: string
  kind?: WorkflowCategoryKind
}

/** One entry in the Workflows popup. Counts come from the catalogue, not from this install. */
export interface WorkflowSummary {
  slug: string
  title: string
  summary: string
  tags: string[]
  heroUrl: string | null
  heroType: 'image' | 'video'
  heroPosterUrl?: string | null
  categories: string[]
  categoryNames: string[]
  viewCount: number
  downloadCount: number
  importCount: number
  shareCount: number
  nodeCount: number
  modelCount: number
  inputCount: number
  publishedAt: string | null
  updatedAt: string | null
  /** The write-up on omnichar.org. On the card so the link costs no detail fetch. */
  pageUrl?: string
}

export interface WorkflowCatalogue {
  entries: WorkflowSummary[]
  categories: WorkflowCategory[]
  /** True when the site was unreachable and these entries are the saved copy. */
  stale: boolean
}

/** A node the workflow builds. Served by the catalogue; the app does not draw it. */
export interface WorkflowNodeSummary {
  type: string
  title: string
  count: number
  source: string
}

/** A weight the workflow declares. Display-only: downloads are driven by the app's own reading. */
export interface WorkflowModelSummary {
  key: string
  label: string
  kind: string
  url: string
  required: boolean
}

/** A file the graph expects in a Load Assets node. Import never fetches these; loaders land empty. */
export interface WorkflowInputSummary {
  key: string
  item_id: string
  slot: number
  label: string
  kind: string
  filename: string
  required: boolean
}

export interface WorkflowDetail {
  slug: string
  title: string
  summary: string
  tags: string[]
  categories: string[]
  categoryNames: string[]
  heroUrl: string | null
  heroType: 'image' | 'video'
  /** The recipe, rebuilt onto the canvas by `buildGraphFromRecipe`. */
  graph: unknown
  nodes: WorkflowNodeSummary[]
  models: WorkflowModelSummary[]
  inputs: WorkflowInputSummary[]
  creator: { name?: string; handle?: string; url?: string } | null
  viewCount: number
  downloadCount: number
  importCount: number
  publishedAt: string | null
  pageUrl: string
  stale?: boolean
}

/** How the Workflows popup orders its cards. Mirrors the catalogue API's `sort` values. */
export type WorkflowSort = 'views' | 'downloads' | 'newest'

/** Core → renderer: how far a reference sweep has got. */
export interface TuneProgressEvent {
  runId: string
  done: number
  total: number
  fraction: number
}

/** What a sweep measured. Read by the node face, its panel, and the HTML report alike. */
export interface SweepResult {
  id: string
  status: string
  error: string
  done: number
  total: number
  warnings: string[]
  character: string
  target: string
  refs: number
  /** Which gallery answered: the frozen originals, or the current references. Not the same claim. */
  gallery: string
  report: string
  spent: number
  /** The one line the node face shows when it finishes. */
  headline: string
  /** Never dropped: a polished panel is where "face only" stops being obvious. */
  caveat: string
  combinations: { key: string; adjusted: number; mean: number; spread: number; tested: number }[]
  contributions: {
    index: number
    delta: number
    spread: number
    pairs: number
    verdict: 'keep' | 'neutral' | 'consider-removing' | 'keeps-the-wardrobe'
    perPrompt: Record<string, number>
    framingShift: number
    /** The same paired difference on the clothes, or null where they were never in frame. */
    wardrobeDelta: number | null
    wardrobePairs: number
  }[]
  /** Only the references the sweep flagged, re-measured on seeds it never rendered. */
  confirmations?: {
    index: number
    initial: number
    retest: number
    spread: number
    pairs: number
    pooled: number
    pooledPairs: number
    agrees: boolean
    verdict: 'removal confirmed' | 'not confirmed' | 'reversed' | 'not re-tested'
  }[]
  cells: {
    combination: string
    prompt: string
    seed: number
    score: number | null
    path: string
    faceFraction: number | null
    /** Null where the clothes were never in frame, which is not the same as a bad match. */
    wardrobe: number | null
  }[]
}

/** Core → renderer: a sweep reached a terminal state, and what it found. */
export interface TuneDoneEvent {
  runId: string
  status: string
  result: SweepResult
}
