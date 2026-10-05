import { useEffect, useState } from 'react'
import { type NodeProps } from '@xyflow/react'
import { isModelPort } from '@shared/coreNodes'
import { isExtensionNode, extensionOf } from '@shared/extensions'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { useGenerationStore } from '../../../store/generationStore'
import { useGraphSelectionStore } from '../../../store/graphSelectionStore'
import { useAssetStore } from '../../../store/assetStore'
import { useFrameStore } from '../../../store/frameStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { activeDownload, useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { useExtensionsStore } from '../../../store/extensionsStore'
import { useLightboxStore } from '../../../store/lightboxStore'
import { matchControlAspect } from '../../../lib/matchControlAspect'
import { resolveCoreInputThumbs } from './coreInputThumbs'
import { CoreOutputPreview, CoreOutputThumb } from './CoreOutputPreview'
import type { SlotId } from './takeSlots'
import {
  applyableParams,
  buildSlots,
  activePending,
  hasEdits,
  restorableKeys,
  slotMedia,
  slotPrompt,
  slotRecipe,
} from './takeSlots'
import type { CorePendingRun, CoreTakeRef } from '@shared/types'
import { NodeFrame } from './NodeFrame'
import { bottomStyle, compactNodeMinHeight, topStyle } from './nodeSize'
import { PortHandle } from './PortHandle'
import { ReferenceStrip } from './ReferenceStrip'
import { scoreSuffix, scoreTone, scoreTitle } from '@/lib/continuity'
import { NodeRunToolbar } from './NodeRunToolbar'
import { missingInputs, missingInputsMessage, optionsWithPick } from '../missingInputs'
import { useGraphMenu } from './useGraphMenu'
import {
  AdjustIcon,
  AlertIcon,
  BoxIcon,
  DownloadIcon,
  ImageGlyph,
  NodeBadge,
  NodeBadgeRow,
  SparkleIcon,
  SquareIcon,
  TypeIcon,
  WandIcon,
  RunningDots,
  StopIcon,
  PencilIcon,
} from './NodeBadge'
import { resolveMedia } from '@/lib/media'
import { SweepResults } from './SweepResults'

interface GraphNodeData extends Record<string, unknown> {
  itemId: string
}

// Handles are packed against an edge rather than spread down the whole side: content/signal ports
// stack from the top, model-family ports (model/vae/text-encoder) stack from the bottom - so model
// wiring reads as one band along the bottom and the image flow runs across the top.
/** One colored port dot with a hover chip naming the port - input (left) or output (right). */
/** Map a Core descriptor's `icon` string to a node-family glyph (falls back to the square). */
function coreGlyph(icon: string): React.JSX.Element {
  switch (icon) {
    case 'wand':
      return <WandIcon />
    case 'box':
      return <BoxIcon />
    case 'type':
      return <TypeIcon />
    case 'image':
      return <ImageGlyph />
    case 'sparkles':
      return <SparkleIcon />
    default:
      return <SquareIcon />
  }
}

/**
 * A generic Inline Core graph node backed by a `core` moodboard item. Resolves its descriptor from
 * the served `/v1/models` palette and renders in the same card style as the fal Generate node: a
 * floating title badge, an edge-to-edge output preview, and a footer with Run + an adjust (settings)
 * button. Params live behind the adjust button in the Core settings sidebar - the node face stays
 * clean, so a model node like Z-Image Turbo reads as one simple node. One colored handle per
 * declared port (inputs left, outputs right, colored by kind).
 */
export function GraphNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { itemId } = data as GraphNodeData
  const item = useMoodboardStore((s) => s.items.find((i) => i.id === itemId))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const setConnectedPromptText = useMoodboardStore((s) => s.setConnectedPromptText)
  const connectors = useMoodboardStore((s) => s.connectors)
  const items = useMoodboardStore((s) => s.items)
  const assets = useAssetStore((s) => s.assets)
  const frames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const openLightbox = useLightboxStore((s) => s.open)
  // Not persisted: a saved browse position reopens a project showing history, not the present.
  const [slot, setSlot] = useState<SlotId>('current')
  // Follow each render as it lands: Core promotes a finished take to the node's active output.
  const activeTakeId = item?.data.core?.output?.takeId
  useEffect(() => {
    if (activeTakeId) setSlot(activeTakeId)
  }, [activeTakeId])
  // What Generate will actually send, so the front slot never shows an older take's prompt.
  const livePrompt = useMoodboardStore((s) => s.connectedPromptText(itemId))
  const coreType = item?.type === 'core' ? item.data.core?.type : undefined
  const descriptor = useCoreNodesStore((s) =>
    coreType ? s.descriptors.find((d) => d.type === coreType) : undefined,
  )
  const runWorkflow = useGenerationStore((s) => s.runWorkflow)
  const cancel = useGenerationStore((s) => s.cancel)
  const toggleSettings = useGenerationStore((s) => s.toggleCoreSettings)
  // This node is the selected graph's output node → it floats the graph's single Run control.
  const isRunTarget = useGraphSelectionStore((s) => s.runTargets.includes(itemId))
  const busy = useGenerationStore((s) => s.busyByFrame[itemId] ?? false)
  // Green follows the node actually executing, not only the one Run was pressed on; red stays on
  // whichever node stopped the run, which in a chain is rarely the same node.
  const executing = useGenerationStore((s) => s.runningNode === itemId)
  const failed = useGenerationStore((s) => s.failedNode === itemId)
  const progress = useGenerationStore((s) => s.progressByFrame[itemId])
  const status = useGenerationStore((s) => s.statusByFrame[itemId])
  const graphMenu = useGraphMenu(itemId, descriptor?.title ?? 'graph')

  // Model requirements (per node type) drive the blinking "missing models" hint + its popup, and
  // surface an in-node download indicator. Refetched when the model registry version changes (a
  // dropped-in file, or a completed download).
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const reqs = useModelRequirementsStore((s) => (coreType ? s.byType[coreType] : undefined))
  const downloadsForType = useModelRequirementsStore((s) =>
    coreType ? s.downloads[coreType] : undefined,
  )
  useEffect(() => {
    if (coreType) void loadReqs(coreType)
  }, [coreType, registryVersion, loadReqs])

  // An installed-but-disabled extension that declares this node type, so the fallback card can say
  // "turn it back on" instead of the generic "not registered".
  const disabledPack = useExtensionsStore((s) =>
    coreType
      ? (s.extensions.find((e) => e.nodes.some((n) => n.type === coreType))?.name ?? null)
      : null,
  )
  const openExtensions = useExtensionsStore((s) => s.openDialog)

  if (!item || item.type !== 'core' || !item.data.core || !descriptor) {
    return (
      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={92}
        subtleSelect
        running={busy || executing}
      >
        <div className="flex h-full flex-col items-center justify-center gap-1 p-3 text-center">
          {coreType && disabledPack ? (
            // The extension is installed but off, so this is a toggle away from working.
            <>
              <span className="text-[11px] font-semibold text-amber-300">Extension disabled</span>
              <span className="text-[10px] leading-tight text-zinc-400">
                <span className="text-zinc-300">{coreType}</span> comes from{' '}
                <span className="text-zinc-300">{disabledPack}</span>, which is turned off.
              </span>
              <button
                onClick={() => openExtensions('installed')}
                className="mt-1 rounded-md border border-border px-2 py-1 text-[10px] font-medium text-zinc-200 hover:border-emerald-500/50 hover:text-emerald-300"
              >
                Open Extensions
              </button>
            </>
          ) : coreType ? (
            <>
              <span className="text-[11px] font-semibold text-amber-300">Node unavailable</span>
              <span className="text-[10px] leading-tight text-zinc-400">
                <span className="text-zinc-300">{coreType}</span> is not registered. Start Inline
                Core and install its runtime (the <span className="text-zinc-300">zimage</span>{' '}
                extra).
              </span>
            </>
          ) : (
            <span className="text-[11px] text-zinc-500">Core node</span>
          )}
        </div>
      </NodeFrame>
    )
  }

  const core = item.data.core
  const pct = typeof progress === 'number' ? Math.round(progress * 100) : null

  // A graph imported from JSON names files the author had; flag the ones this machine does not.
  const missing = missingInputs(descriptor, core.params)
  const missingMessage = missingInputsMessage(missing)

  // Take history for the on-node output strip (newest first). Older items predate history and only
  // carry a single `output` - treat that as a one-entry history. `output` marks the active take.
  // Minus the installed-file dropdowns: a take records the runner's `model`, not the filename.
  const restorable = restorableKeys(descriptor?.params)
  const edited = hasEdits(core, livePrompt)
  const slots = buildSlots(core, busy || executing, edited)
  // A slot that has gone falls back to the active output rather than highlighting nothing.
  const shown = slots.some((e) => e.id === slot) ? slot : (core.output?.takeId ?? 'current')

  const shownPrompt = slotPrompt(core, shown, livePrompt)
  const shownMedia = slotMedia(core, shown)
  const rendering = (busy || executing) && shown === 'current'
  // Selecting a slot restores the graph that produced it, seed excluded: a pinned seed turns on
  // the node cache and freezes re-generation.
  const restore = (
    recipe: { params?: Record<string, unknown>; prompt?: string } | undefined,
    output?: CoreTakeRef,
    pending?: CorePendingRun,
  ): void => {
    const params = recipe ? { ...core.params, ...applyableParams(recipe, restorable) } : core.params
    const next = { ...core, params }
    if (output) next.output = output
    if (pending) next.pending = pending
    void updateItem(itemId, { data: { ...item.data, core: next } })
    if (typeof recipe?.prompt === 'string') void setConnectedPromptText(itemId, recipe.prompt)
  }

  const selectTake = (o: CoreTakeRef): void => {
    // Settings edited but never generated have no take to come back to, and this restore is about
    // to overwrite them. Captured here rather than on every keystroke: this is the only moment they
    // can be lost, so it is the only moment worth a write. It replaces a stopped run's snapshot,
    // which the edit has superseded; it never replaces a running one, whose settings are the only
    // record of what is on the GPU right now.
    // A snapshot must exist before this overwrites the node's params. Never over a draft or a
    // running run: see "the draft survives browsing" in docs/generation-recipe.md. A stopped run's snapshot is replaced only by a real edit.
    const held = activePending(core)?.status
    const draft: CorePendingRun | undefined =
      held === 'running' || held === 'draft' || (held !== undefined && !edited)
        ? undefined
        : { params: { ...core.params }, prompt: livePrompt, startedAt: Date.now(), status: 'draft' }
    setSlot(o.takeId)
    restore(o, o, draft)
  }

  const selectCurrent = (): void => {
    setSlot('current')
    restore(slotRecipe(core, 'current'))
  }

  // Real "models missing" signal from the requirements check (replaces the old options heuristic,
  // which wrongly assumed a silent auto-download). Assume OK until requirements load, to avoid a
  // hint flash on first render.
  const modelsMissing = reqs ? !reqs.allPresent : false
  const download = downloadsForType ? activeDownload(downloadsForType, reqs) : null
  const downloadPct = download ? Math.round(download.fraction * 100) : null
  // Apply ControlNet's detector download is scoped to the selected type, so canny never nags for a
  // model and depth/pose only prompt for the one they use (component ids start with the detector).
  const applyType = coreType === 'control/apply' ? String(core?.params?.type ?? 'pose') : null
  const detectorNoun = applyType === 'depth' ? 'MiDaS' : applyType === 'pose' ? 'OpenPose' : null
  const detectorPrefix = applyType === 'depth' ? 'midas' : applyType === 'pose' ? 'openpose' : null
  // A suggested (optional) component that isn't on disk yet - the opt-in ControlNet, or (for Apply
  // ControlNet) the detector its selected type needs. Surfaced as a soft chip, not the "missing" alarm.
  const suggested = applyType
    ? detectorPrefix
      ? (reqs?.components.find((c) => !c.present && c.id.startsWith(detectorPrefix)) ?? null)
      : null // canny needs no model
    : (reqs?.components.find((c) => c.optional && !c.present) ?? null)
  const suggestedDl = suggested && downloadsForType ? downloadsForType[suggested.id] : undefined
  const suggestedPct = suggestedDl ? Math.round(suggestedDl.fraction * 100) : null
  // The chip's noun: the detector for Apply ControlNet, else the opt-in ControlNet model.
  const suggestNoun =
    detectorNoun ?? (suggested?.category === 'annotators' ? 'detectors' : 'ControlNet')

  // Offer "Match aspect" only when a control map is actually wired into this node's Control input.
  const controlWired =
    (descriptor.inputs?.some((p) => p.id === 'control_image') ?? false) &&
    connectors.some(
      (c) =>
        c.toItemId === itemId &&
        (c.data as { targetHandle?: string }).targetHandle === 'control_image',
    )

  // A node with no media output renders compact: no preview card, since there is nothing to show.
  const isLoader = descriptor.outputKind == null
  // Run belongs on whatever ends the graph. A media node always ends one; a node with no media
  // output ends it only when nothing downstream consumes it, which is what separates Write .char
  // (the end of a character graph) from a loader feeding the node that will run it.
  // A display reads a run; it does not run one. Counting it as a consumer took the Run control
  // off the sweep node the moment a Logger was wired to it, leaving no way to start it.
  const feedsSomething = connectors.some(
    (c) =>
      c.fromItemId === itemId &&
      !DISPLAY_ONLY.has(items.find((i) => i.id === c.toItemId)?.type ?? ''),
  )
  const showRun = !isLoader || !feedsSomething
  const fileParam = core?.params?.file
  // Name the weights rather than saying "Auto": the point of the label is to tell you what will
  // load. A generation node's provider resolves this; a plain loader has none, so fall back to the
  // first file in its catalog, which is the one the engine auto-picks anyway.
  const fileField = descriptor.params.find((p) => p.key === 'file')
  // Load Character never auto-picks, so naming the first catalog entry showed a pick Core never got.
  const picksCharacter = fileField?.optionsFrom === 'characters'
  const fileFallback = picksCharacter
    ? 'No character picked'
    : fileField?.default || fileField?.options?.[0]?.value
  const fileLabel = fileField ? String(fileParam || fileFallback || 'Not installed') : ''
  // An extension-provided node carries its owning extension's id (`ext:<id>:<module>`) - surface it
  // as a chip so it's clear which extension a canvas node came from.
  const extName = isExtensionNode(descriptor.source) ? extensionOf(descriptor.source) : null

  // The stored value is the `.char` filename; the badge shows the name without the suffix.
  const character = String(core.params?.character ?? '').replace(/\.char$/i, '')

  // A reference list (FLUX.2 and friends): the prompt addresses these by position, "the jacket from
  // image 2", so the card numbers them in wiring order - the same order graph_build sends the engine.
  const listPort = descriptor.inputs.find((p) => p.kind === 'image[]')
  const references = listPort
    ? resolveCoreInputThumbs(itemId, listPort.id, {
        items,
        connectors,
        assets,
        frames,
        takesByFrame,
      })
    : []

  // Split each side into content (top-packed) and model-family (bottom-packed) ports.
  const inContent = descriptor.inputs.filter((p) => !isModelPort(p.kind))
  const inModel = descriptor.inputs.filter((p) => isModelPort(p.kind))
  const outContent = descriptor.outputs.filter((p) => !isModelPort(p.kind))
  const outModel = descriptor.outputs.filter((p) => isModelPort(p.kind))

  const handles = (
    <>
      {inContent.map((port, i) => (
        <PortHandle
          key={port.id}
          id={port.id}
          label={port.label}
          kind={port.kind}
          side="input"
          style={topStyle(i)}
        />
      ))}
      {inModel.map((port, i) => (
        <PortHandle
          key={port.id}
          id={port.id}
          label={port.label}
          kind={port.kind}
          side="input"
          style={bottomStyle(i)}
        />
      ))}
      {outContent.map((port, i) => (
        <PortHandle
          key={port.id}
          id={port.id}
          label={port.label}
          kind={port.kind}
          side="output"
          style={topStyle(i)}
        />
      ))}
      {outModel.map((port, i) => (
        <PortHandle
          key={port.id}
          id={port.id}
          label={port.label}
          kind={port.kind}
          side="output"
          style={bottomStyle(i)}
        />
      ))}
    </>
  )

  const runToolbar = showRun ? (
    <NodeRunToolbar
      isTarget={isRunTarget}
      busy={busy}
      onRun={() => void runWorkflow(itemId)}
      onStop={() => void cancel(itemId)}
      disabled={download !== null}
      disabledReason="Downloading model…"
      menuItems={graphMenu.items}
      menuNote={graphMenu.note}
    />
  ) : null

  if (isLoader) {
    // A loader's whole job is picking a file, so its SELECT dropdown(s) live directly on the node
    // face (not behind Adjust) - the one exception to "params off the node face", which exists to
    // keep *generation* one-click. Any non-select params (rare for a loader) stay behind Adjust.
    // A select has always sat on the face; anything else opts in with `onFace`. Adjust still
    // lists every param, so the face is a shortcut to the ones worth seeing, never the only way in.
    const onFace = (field: (typeof descriptor.params)[number]): boolean =>
      field.onFace ?? field.widget === 'select'
    const faceParams = descriptor.params.filter(onFace)
    const otherParams = descriptor.params.filter((p) => !onFace(p))
    // Numbers store as numbers: the recipe declares each param's type, and a string under
    // `type: "number"` leaves a reader guessing. Coerced only when it round-trips, so a half-typed
    // "0." survives long enough to become "0.5".
    const setParam = (key: string, value: string, widget?: string): void => {
      const trimmed = value.trim()
      const stored: string | number =
        widget === 'number' && trimmed !== '' && String(Number(trimmed)) === trimmed
          ? Number(trimmed)
          : value
      void updateItem(itemId, {
        data: { ...item.data, core: { ...core, params: { ...core.params, [key]: stored } } },
      })
    }
    return (
      <>
        {runToolbar}
        <NodeBadgeRow dragNodeId={id}>
          <NodeBadge icon={coreGlyph(descriptor.icon)}>{descriptor.title}</NodeBadge>
          {extName && (
            <NodeBadge tone="info" title={`From the ${extName} extension`}>
              {extName}
            </NodeBadge>
          )}
          {/* A loader can also declare downloadable weights (e.g. an extension's `models`). It has
              no preview overlay to host the download state, so surface it on the title badge. */}
          {(modelsMissing || download) && (
            <button
              onClick={() => openReqs(descriptor.type)}
              title={download ? 'Downloading model…' : 'Models missing - click to download'}
              className={`nodrag flex h-6 items-center gap-1 rounded-full border px-2 text-[10px] font-medium shadow-sm backdrop-blur ${
                download
                  ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                  : 'animate-pulse border-amber-500/40 bg-amber-500/10 text-amber-300 hover:animate-none hover:bg-amber-500/20'
              }`}
            >
              <AlertIcon className="h-3.5 w-3.5" />
              {download ? `${downloadPct}%` : 'Models'}
            </button>
          )}
        </NodeBadgeRow>
        <NodeFrame
          id={id}
          selected={!!selected}
          minWidth={188}
          minHeight={compactNodeMinHeight(descriptor)}
          padded={false}
          subtleSelect
          running={busy || executing}
          invalid={missing.length > 0 || failed}
        >
          <div className="flex h-full w-full flex-col gap-1 px-2 py-1.5">
            {faceParams.length > 0 ? (
              faceParams.map((field) => {
                if (field.widget === 'textarea') {
                  // A one-line input would hide nine of ten prompts, which is the whole param.
                  const held = core.params?.[field.key]
                  return (
                    <FaceField key={field.key} label={field.label}>
                      <textarea
                        key={field.key}
                        rows={4}
                        value={String(held ?? field.default ?? '')}
                        placeholder={field.label}
                        title={field.label}
                        onChange={(e) => setParam(field.key, e.target.value, field.widget)}
                        className="nodrag nowheel w-full min-w-0 resize-none rounded border border-border bg-panel px-1.5 py-1 font-mono text-[10px] leading-snug text-zinc-200 outline-none focus:border-accent"
                      />
                    </FaceField>
                  )
                }
                if (field.widget !== 'select') {
                  // Unset shows the default the run will actually use; cleared stays cleared. The
                  // two are distinguishable, and a box reading empty beside a strength of 1 is the
                  // node claiming it has no value.
                  const held = core.params?.[field.key]
                  const isNumber = field.widget === 'number'
                  return (
                    <FaceField key={field.key} label={field.label}>
                      <input
                        type={isNumber ? 'number' : 'text'}
                        min={isNumber ? field.min : undefined}
                        max={isNumber ? field.max : undefined}
                        step={isNumber ? field.step : undefined}
                        value={String(held ?? field.default ?? '')}
                        placeholder={field.label}
                        title={field.label}
                        onChange={(e) => setParam(field.key, e.target.value, field.widget)}
                        className="nodrag w-full min-w-0 rounded border border-border bg-panel px-1.5 py-1 text-[10px] text-zinc-200 outline-none focus:border-accent"
                      />
                    </FaceField>
                  )
                }
                const opts = field.options ?? []
                const hasAuto = opts.some((o) => o.value === '')
                // A node saved before Core resolved these still stores "", which matches no option
                // and renders blank; fall back to the resolved default.
                const stored = core.params?.[field.key]
                // Empty means "engine auto-picks", which is the first file; show that rather than a
                // blank select. Display only - the stored value stays empty until the user picks.
                // Not for a character: Core never auto-picks one, and showing the first as chosen
                // meant picking it fired no change, so it could never be saved.
                const fallback =
                  field.optionsFrom === 'characters' ? '' : field.default || opts[0]?.value || ''
                const selected = stored == null || stored === '' ? fallback : stored
                // A pick the catalog lacks stays in the list, or the select renders blank and the
                // name the graph arrived with is gone.
                const shown = optionsWithPick(opts, String(selected))
                return (
                  <FaceField key={field.key} label={field.label}>
                    <select
                      value={String(selected)}
                      onChange={(e) => setParam(field.key, e.target.value)}
                      title={field.label}
                      className="nodrag w-full min-w-0 rounded border border-border bg-panel px-1.5 py-1 text-[10px] text-zinc-200 outline-none focus:border-accent"
                    >
                      {/* Only when nothing resolved: otherwise the concrete file is already shown. */}
                      {!hasAuto && !field.default && (
                        <option value="">{`Select ${field.label}`}</option>
                      )}
                      {shown.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </FaceField>
                )
              })
            ) : fileLabel ? (
              <span
                className="min-w-0 flex-1 truncate px-1 font-mono text-[10px] text-zinc-400"
                title={fileLabel}
              >
                {fileLabel}
              </span>
            ) : (
              <span className="min-w-0 flex-1 truncate px-1 text-[10px] text-zinc-500">
                {descriptor.title}
              </span>
            )}
            {/* A sweep publishes its own progress and verdict; every other loader has none. */}
            {descriptor.type === 'character/finetune' && (
              <SweepResults runId={(item.data.runId as string | null | undefined) ?? null} />
            )}
            {otherParams.length > 0 && (
              <div className="flex justify-end">
                <button
                  onClick={() => toggleSettings(itemId)}
                  title="Settings"
                  data-gen-settings-toggle
                  className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
                >
                  <AdjustIcon />
                </button>
              </div>
            )}
          </div>
        </NodeFrame>
        {handles}
      </>
    )
  }

  return (
    <>
      {/* The graph's single Run control, floated above this output node while the graph is selected. */}
      {runToolbar}
      {/* Floating title badge - matches the fal Generate node. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={coreGlyph(descriptor.icon)}>{descriptor.title}</NodeBadge>
        {extName && (
          <NodeBadge tone="info" title={`From the ${extName} extension`}>
            {extName}
          </NodeBadge>
        )}
        {/* Params live behind Adjust, but an applied character changes what the node generates,
            so it has to be readable without opening the sidebar. */}
        {character && (
          <NodeBadge
            tone="info"
            accent="text-sky-300"
            title={`Generating with the character ${character}`}
          >
            {character}
          </NodeBadge>
        )}
        {/* A pick the catalog does not have, e.g. the LoRA an imported graph was authored with.
            Distinct from `modelsMissing`, which is about a node type's own downloadable weights. */}
        {missing.length > 0 && (
          <NodeBadge tone="info" accent="text-red-400" title={missingMessage}>
            {missing.length === 1 ? missing[0].label : `${missing.length} inputs`} missing
          </NodeBadge>
        )}
        {modelsMissing && (
          <button
            onClick={() => openReqs(descriptor.type)}
            title="Models missing - click to download"
            className="nodrag flex h-6 animate-pulse items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2 text-[10px] font-medium text-amber-300 shadow-sm backdrop-blur hover:animate-none hover:bg-amber-500/20"
          >
            <AlertIcon className="h-3.5 w-3.5" />
            Models
          </button>
        )}
        {suggested && !modelsMissing && (
          <button
            onClick={() => openReqs(descriptor.type)}
            title={`${suggested.label} - optional download`}
            className="nodrag flex h-6 items-center gap-1 rounded-full border border-border bg-panel/80 px-2 text-[10px] font-medium text-zinc-300 shadow-sm backdrop-blur hover:border-emerald-500/40 hover:text-emerald-300"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            {suggestedDl && !suggestedDl.error
              ? `${suggestNoun} ${suggestedPct}%`
              : `Get ${suggestNoun}`}
          </button>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={200}
        padded={false}
        subtleSelect
        running={busy || executing}
        invalid={missing.length > 0 || failed}
      >
        <div className="relative flex h-full w-full flex-col">
          {/* Edge-to-edge output preview. */}
          <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
            {references.length > 0 && <ReferenceStrip references={references} />}
            {shownMedia ? (
              // Dimmed while rendering rather than blanked: an empty card through a long render
              // reads as broken, and the previous take is the most useful thing to look at.
              <div
                className={
                  rendering ? 'h-full w-full opacity-40 transition-opacity' : 'h-full w-full'
                }
              >
                <CoreOutputPreview
                  filePath={shownMedia.filePath}
                  kind={shownMedia.kind}
                  name={shownMedia.prompt || descriptor.title}
                  onExpand={(kind) =>
                    openLightbox({
                      src: resolveMedia(shownMedia.filePath),
                      kind,
                      name: shownMedia.prompt || descriptor.title,
                    })
                  }
                />
              </div>
            ) : (
              <div className="flex h-full w-full items-center justify-center px-4">
                <span className="text-center text-[10px] text-zinc-600">
                  {busy
                    ? (status ?? 'Working…')
                    : download
                      ? `Downloading ${download.label}…`
                      : modelsMissing
                        ? 'Models missing - click the hint to download'
                        : 'Run to generate'}
                </span>
              </div>
            )}

            {/* Busy = generating; else show a download indicator while a model is being fetched. */}
            {busy ? (
              <>
                <span className="absolute left-2 top-2 flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-emerald-300 backdrop-blur">
                  <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
                  <span className="truncate">
                    {status ?? (pct != null ? `${pct}%` : 'Working…')}
                  </span>
                </span>
                <div className="absolute inset-x-0 bottom-0 h-1 bg-black/40">
                  <div
                    className="h-full bg-emerald-400 transition-all"
                    style={{ width: `${pct ?? 0}%` }}
                  />
                </div>
              </>
            ) : (
              download && (
                <>
                  <span className="absolute left-2 top-2 flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-sky-300 backdrop-blur">
                    <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-sky-400" />
                    <span className="truncate">
                      {download.label} {downloadPct != null ? `${downloadPct}%` : ''}
                    </span>
                  </span>
                  <div className="absolute inset-x-0 bottom-0 h-1 bg-black/40">
                    <div
                      className="h-full bg-sky-400 transition-all"
                      style={{ width: `${downloadPct ?? 0}%` }}
                    />
                  </div>
                </>
              )
            )}
          </div>

          {/* Take history behind a permanent Current slot. Current holds the node's live settings,
              which is what makes browsing reversible: selecting it again is how you get back. */}
          <div className="nowheel flex shrink-0 gap-1 overflow-x-auto border-t border-border bg-surface/90 px-1.5 py-1.5">
            {slots.map((entry) => {
              const take = entry.take
              const active = entry.id === shown
              const ring = active
                ? 'border-emerald-400 ring-1 ring-emerald-400/40'
                : 'border-border hover:border-zinc-500'
              if (!take) {
                // One slot for everything that is not a finished render: edited, running, stopped.
                const face = {
                  draft: {
                    tone: 'border-dashed border-zinc-500 hover:border-zinc-400',
                    title: 'Edited since the last render - click to restore these settings',
                    glyph: <PencilIcon className="h-4 w-4 text-zinc-400" />,
                  },
                  failed: {
                    tone: 'border-red-500/70 hover:border-red-400',
                    title: 'Failed - click to restore the settings it ran with',
                    glyph: <StopIcon className="h-4 w-4 text-red-400" />,
                  },
                  cancelled: {
                    tone: 'border-amber-500/70 hover:border-amber-400',
                    title: 'Cancelled - click to restore the settings it ran with',
                    glyph: <StopIcon className="h-4 w-4 text-amber-400" />,
                  },
                  running: {
                    tone: ring,
                    title: `Rendering${pct === null ? '' : ` ${pct}%`} - click to restore its settings`,
                    // Dots mean working, so no other state may wear them.
                    glyph: <RunningDots className="text-emerald-300" />,
                  },
                }[entry.state === 'take' ? 'draft' : entry.state]
                return (
                  <button
                    key="current"
                    onClick={selectCurrent}
                    title={face.title}
                    className={`nodrag relative flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded border bg-black/60 transition-colors ${active ? ring : face.tone}`}
                  >
                    {face.glyph}
                    {entry.state === 'running' && (
                      <span className="absolute inset-x-0 bottom-0 h-0.5 bg-emerald-400/70" />
                    )}
                  </button>
                )
              }
              return (
                <button
                  key={entry.id}
                  onClick={() => selectTake(take)}
                  title={active ? 'Shown' : 'Use this take'}
                  className={`nodrag relative h-11 w-11 shrink-0 overflow-hidden rounded border transition-colors ${ring}`}
                >
                  <CoreOutputThumb filePath={take.filePath} kind={take.kind} />
                  {/* Only when a character was applied and a score was actually measured: an
                      unmeasurable take must not read as a zero-scoring one. */}
                  {take.continuityScore !== undefined && (
                    <span
                      title={scoreTitle(
                        take.continuityScore,
                        take.continuityFaceOnly,
                        take.continuitySubjectOnly,
                        take.continuityWardrobe,
                      )}
                      className={`pointer-events-none absolute left-0.5 top-0.5 rounded bg-black/85 px-1 text-[8px] leading-tight ${scoreTone(take.continuityScore)}`}
                    >
                      {Math.round(take.continuityScore)}
                      {scoreSuffix(take.continuityFaceOnly, take.continuitySubjectOnly)}
                    </span>
                  )}
                </button>
              )
            })}
          </div>

          {/* The selected slot's prompt: the browsed take's, or the live one on Current. It used to
              always show the active take's, so an edited prompt left the node advertising the old. */}
          {shownPrompt && (
            <div
              className="shrink-0 truncate border-t border-border bg-surface/90 px-2 py-1 text-[10px] text-zinc-400"
              title={shownPrompt}
            >
              {shownPrompt}
            </div>
          )}

          {/* Footer: category label + settings (adjust). Run lives on the graph's output node. */}
          <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border bg-surface/90 px-1.5 py-1">
            <span className="truncate px-1 text-[10px] uppercase tracking-wide text-zinc-500">
              {descriptor.category}
            </span>
            <div className="flex shrink-0 items-center gap-1">
              {controlWired && (
                <button
                  onClick={() => void matchControlAspect(itemId)}
                  title="Set Width/Height to the wired control map's aspect so the pose isn't stretched"
                  className="nodrag flex h-6 items-center rounded px-1.5 text-[10px] text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
                >
                  Match aspect
                </button>
              )}
              <button
                onClick={() => toggleSettings(itemId)}
                title="Settings"
                data-gen-settings-toggle
                className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
              >
                <AdjustIcon />
              </button>
            </div>
          </div>
        </div>
      </NodeFrame>

      {handles}
    </>
  )
}

/** Nodes that only show what a run produced, so wiring one never changes what a graph does. */
const DISPLAY_ONLY: ReadonlySet<string> = new Set(['log/tail', 'train/loss'])

/** A face control with its name above it: a placeholder disappears the moment there is a value. */
function FaceField({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <label className="flex min-w-0 flex-col gap-0.5">
      <span className="truncate text-[9px] uppercase tracking-wide text-zinc-500" title={label}>
        {label}
      </span>
      {children}
    </label>
  )
}
