/**
 * The Trainer node's Adjust sidebar - hyperparameters live here, never on the node face (the same
 * split the Generate / Inline Core nodes use).
 *
 * Edits are staged, not live. A checkpoint encodes the rank, targets and base it was built with, so
 * silently applying a change would leave a Resume that trains something other than what the panel
 * says. Instead the edits collect behind an Update button, and applying them to a node with a
 * resumable run asks first, then discards that run's checkpoints.
 */
import { useEffect, useState } from 'react'
import { clipGridFor, resolveClipLengthFor } from '@shared/clipGrid'

import type {
  TrainingArch,
  TrainingBaseMode,
  TrainingBaseQuant,
  TrainingHyperparams,
  TrainingLoraScope,
  TrainingMode,
  TrainingOffload,
} from '@shared/types'
import { Modal } from '../../components/Modal'
import { useBoardActions } from '../Moodboard/nodes/boardActions'
import { useTrainingStore } from '../../store/trainingStore'
import { XIcon } from '../Moodboard/nodes/NodeBadge'

const DEFAULTS: TrainingHyperparams = {
  arch: 'z-image',
  baseMode: 'deturbo',
  baseQuant: 'auto',
  offload: 'auto',
  rank: 16,
  alpha: 16,
  learningRate: 1e-4,
  steps: 1500,
  batchSize: 1,
  resolution: 1024,
  saveEvery: 250,
  saveSnapshots: false,
  gpuIds: [],
  outputName: '',
  loraScope: 'full',
  captionDropout: 0.05,
  flipAugment: false,
  clipSeconds: 1,
}

/** Base checkpoints per architecture, recommended first. */
const BASES: Record<TrainingArch, { value: TrainingBaseMode; label: string }[]> = {
  'z-image': [
    { value: 'deturbo', label: 'Z-Image De-Turbo' },
    { value: 'turbo_adapter', label: 'Z-Image Turbo (+ training adapter)' },
  ],
  krea2: [
    { value: 'raw', label: 'Krea 2 RAW (recommended)' },
    { value: 'turbo_adapter', label: 'Krea 2 Turbo (+ training adapter)' },
  ],
  // dev is guidance-distilled rather than step-distilled, so it is itself the training base and
  // there is no adapter and no base/turbo split. schnell is step-distilled and is refused.
  flux1: [{ value: 'raw', label: 'FLUX.1 dev (required)' }],
  // FLUX.2 has no de-distillation adapter: you train on a Base checkpoint and the adapter still
  // loads on the distilled build for generation, which is both faster and better.
  flux2: [
    { value: 'raw', label: 'FLUX.2 klein Base 4B (recommended)' },
    { value: 'raw_9b', label: 'FLUX.2 klein Base 9B (needs ~22GB)' },
  ],
  // H3 ships one undistilled build per partition, and only fl2va trains. The LoRA loads on all four
  // H3 nodes afterwards, ref2va included - the two partitions are the same architecture.
  'minimax-h3': [{ value: 'raw', label: 'MiniMax H3 FL2VA (required)' }],
  // LTX-2.5 trains on the dev transformer. The adapter then loads on all three LTX nodes, in fast
  // mode as well as quality - they are the same architecture.
  'ltx-2-5': [{ value: 'raw', label: 'LTX-2.5 dev (required)' }],
  // int8 ConvRot builds first: the 5.4GB base trains in about 6GB and is the file generation runs.
  'moss-image': [
    { value: 'sft', label: 'MOSS-Image SFT, int8 (recommended)' },
    { value: 'sft_bf16', label: 'MOSS-Image SFT, bf16' },
    { value: 'raw', label: 'MOSS-Image Base, int8' },
    { value: 'raw_bf16', label: 'MOSS-Image Base, bf16' },
  ],
}

/**
 * Which archs offer a base-precision choice. Z-Image is absent because it trains in bf16 on the
 * cards people have; MiniMax H3 is absent for the opposite reason - it is 4-bit only, and a picker
 * with one option is a lie. Both are hidden rather than shown and then refused.
 */
const QUANTIZABLE: TrainingArch[] = ['krea2', 'flux1', 'flux2', 'moss-image']

const QUANTS: { value: TrainingBaseQuant; label: string }[] = [
  { value: 'auto', label: 'Auto (fit to this GPU)' },
  { value: 'none', label: 'Full precision (bf16)' },
  { value: 'nf4', label: '4-bit (NF4)' },
]

/** MOSS-Image quantizes to int8 ConvRot instead: its model cannot run on NF4 weights. */
const MOSS_QUANTS: { value: TrainingBaseQuant; label: string }[] = [
  { value: 'auto', label: 'Auto (fit to this GPU)' },
  { value: 'none', label: 'Full precision (bf16)' },
  { value: 'int8', label: 'int8 ConvRot' },
]

/** Activation offload to host RAM, the way to fit a bf16 base on a card that is a few GB short. */
const OFFLOADS: { value: TrainingOffload; label: string }[] = [
  { value: 'auto', label: 'Auto (only if bf16 will not fit)' },
  { value: 'on', label: 'On (stream to RAM)' },
  { value: 'off', label: 'Off' },
]

const SCOPES: { value: TrainingLoraScope; label: string }[] = [
  { value: 'full', label: 'Full (attention + feed-forward)' },
  { value: 'attention', label: 'Attention only' },
]

function quantsFor(arch: TrainingArch): { value: TrainingBaseQuant; label: string }[] {
  return arch === 'moss-image' ? MOSS_QUANTS : QUANTS
}

const ARCHS: { value: TrainingArch; label: string }[] = [
  { value: 'z-image', label: 'Z-Image' },
  { value: 'krea2', label: 'Krea 2' },
  { value: 'flux1', label: 'FLUX.1' },
  { value: 'flux2', label: 'FLUX.2' },
  { value: 'minimax-h3', label: 'MiniMax H3 (video)' },
  // "(video)" like H3, not "(video + audio)": the model generates a soundtrack, but training only
  // adapts the video branch, and this label sits in the Trainer.
  { value: 'ltx-2-5', label: 'LTX-2.5 (video)' },
  { value: 'moss-image', label: 'MOSS-Image' },
]

function NumberField({
  label,
  value,
  step,
  onChange,
}: {
  label: string
  value: number
  step?: number
  onChange: (v: number) => void
}): React.JSX.Element {
  // Held as text so the field can be empty mid-edit. Binding the number directly turns a backspace
  // into Number('') === 0, which re-renders as "0" and traps you into editing around it.
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])

  return (
    <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
      {label}
      <input
        type="number"
        value={text}
        step={step ?? 1}
        onChange={(e) => {
          setText(e.target.value)
          const parsed = Number(e.target.value)
          if (e.target.value.trim() !== '' && Number.isFinite(parsed)) onChange(parsed)
        }}
        onBlur={() => setText(String(value))}
        className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
      />
    </label>
  )
}

export function TrainerSettingsPanel({ itemId }: { itemId: string }): React.JSX.Element | null {
  const { items, patchData, toggleSettings } = useBoardActions()
  const item = items.find((i) => i.id === itemId)
  const gpus = useTrainingStore((s) => s.systemStats?.gpus) ?? []
  const runs = useTrainingStore((s) => s.runs)
  const discard = useTrainingStore((s) => s.discard)

  const applied: TrainingHyperparams = { ...DEFAULTS, ...(item?.data.hyperparams ?? {}) }
  const [hp, setHp] = useState<TrainingHyperparams>(applied)
  const [confirming, setConfirming] = useState(false)
  // Re-seed when the sidebar moves to another node. Adjusting state during render (rather than in
  // an effect) is React's own guidance for this, and avoids a frame showing the wrong node's values.
  const [seededFor, setSeededFor] = useState(itemId)
  if (seededFor !== itemId) {
    setSeededFor(itemId)
    setHp(applied)
    setConfirming(false)
  }

  const runId = (item?.data.runId as string | null | undefined) ?? null
  const run = runs.find((r) => r.id === runId) ?? null
  // A run holding a checkpoint is the case worth asking about: its checkpoint was built for the
  // old rank/targets/base, so applying new settings means that Resume can no longer be honoured.
  const resumable = run !== null && (run.status === 'interrupted' || run.status === 'failed')
  const dirty = JSON.stringify(hp) !== JSON.stringify(applied)

  if (!item) return null
  // The sidebar is shared by every Trainer node that has settings, so it dispatches on the item.
  // Everything below this point is training hyperparams and only applies to the Trainer node.
  if (item.type === 'train/caption')
    return <CaptionSettings itemId={itemId} overwrite={Boolean(item.data.overwrite)} />

  const arch: TrainingArch = hp.arch ?? 'z-image'
  const set = <K extends keyof TrainingHyperparams>(key: K, value: TrainingHyperparams[K]): void =>
    setHp((current) => ({ ...current, [key]: value }))

  // Base modes and precisions are per-architecture, so a switch must land on values that exist.
  const setArch = (next: TrainingArch): void =>
    setHp((current) => ({
      ...current,
      arch: next,
      baseMode: BASES[next][0].value,
      baseQuant: quantsFor(next).some((q) => q.value === current.baseQuant)
        ? current.baseQuant
        : 'auto',
    }))

  // What the typed seconds actually resolve to on H3's frame grid, shown under the field.
  const clipGrid = clipGridFor(arch)
  const resolved = clipGrid ? resolveClipLengthFor(clipGrid, hp.clipSeconds ?? 1) : null

  const toggleGpu = (index: number): void =>
    set(
      'gpuIds',
      hp.gpuIds.includes(index) ? hp.gpuIds.filter((g) => g !== index) : [...hp.gpuIds, index],
    )

  const commit = async (): Promise<void> => {
    if (resumable && run) {
      await discard(run.id)
      await patchData(itemId, { runId: null })
    }
    await patchData(itemId, { hyperparams: hp })
    setConfirming(false)
  }

  const update = (): void => {
    if (resumable) setConfirming(true)
    else void commit()
  }

  return (
    <div className="absolute right-0 top-0 z-40 flex h-full w-80 flex-col gap-3 overflow-y-auto border-l border-border bg-panel p-4 shadow-2xl">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-200">Training settings</span>
        <button
          onClick={() => toggleSettings(itemId)}
          className="rounded p-1 text-zinc-400 hover:bg-panel hover:text-zinc-200"
          title="Close"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {dirty && (
        <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-2">
          <span className="flex-1 text-[11px] text-amber-200">
            {resumable ? 'Changed. Applying discards the checkpoint.' : 'Changed, not applied yet.'}
          </span>
          <button
            onClick={() => setHp(applied)}
            className="rounded px-2 py-1 text-[11px] text-zinc-400 hover:text-zinc-200"
          >
            Revert
          </button>
          <button
            onClick={update}
            className="rounded-md bg-amber-500/80 px-2.5 py-1 text-[11px] font-medium text-black hover:bg-amber-400"
          >
            Update
          </button>
        </div>
      )}

      <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
        Output LoRA name
        <input
          value={hp.outputName ?? ''}
          placeholder="auto (from the run name)"
          onChange={(e) => set('outputName', e.target.value)}
          className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
        />
        <span className="text-[10px] text-zinc-600">
          Saved as models/loras/&lt;name&gt;.safetensors
        </span>
      </label>

      <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
        Architecture
        <select
          value={arch}
          onChange={(e) => setArch(e.target.value as TrainingArch)}
          className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
        >
          {ARCHS.map((a) => (
            <option key={a.value} value={a.value}>
              {a.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
        Base
        <select
          value={hp.baseMode}
          onChange={(e) => set('baseMode', e.target.value as TrainingBaseMode)}
          className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
        >
          {BASES[arch].map((b) => (
            <option key={b.value} value={b.value}>
              {b.label}
            </option>
          ))}
        </select>
        {arch === 'krea2' && (
          <span className="text-[10px] text-zinc-600">
            Train on RAW, then generate with Krea 2 Turbo - the LoRA carries over.
          </span>
        )}
        {arch === 'flux1' && (
          <span className="text-[10px] text-zinc-600">
            A 24GB bf16 base, so it trains in 4-bit on most cards. dev's weights are non-commercial,
            and a LoRA trained on them inherits that.
          </span>
        )}
        {arch === 'flux2' && (
          <span className="text-[10px] text-zinc-600">
            Train on Base, then generate with the distilled build - the LoRA carries over.
          </span>
        )}
        {arch === 'minimax-h3' && (
          <span className="text-[10px] text-zinc-600">
            Trains on stills, or on short clips if the dataset has any. Wire the result into any H3
            node's LoRA input. 4-bit base, so it needs a big card.
          </span>
        )}
        {arch === 'moss-image' && (
          <span className="text-[10px] text-zinc-600">
            Train on SFT to generate with SFT. An int8 base trains as stored, and the LoRA loads on
            the int8 and bf16 builds alike.
          </span>
        )}
        {arch === 'ltx-2-5' && (
          <span className="text-[10px] text-zinc-600">
            Trains on clips. Wire the result into any LTX-2.5 node's LoRA input. A 22B base, so this
            wants a 48GB card.
          </span>
        )}
      </label>

      {clipGrid && (
        <>
          {arch === 'ltx-2-5' && (
            <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
              Training mode
              <select
                value={hp.trainingMode ?? 'clip'}
                onChange={(e) => set('trainingMode', e.target.value as TrainingMode)}
                className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
              >
                <option value="clip">Clip LoRA (a look and how it moves)</option>
                <option value="control">Control LoRA (reference to target)</option>
              </select>
              <span className="text-[10px] text-zinc-600">
                A Clip LoRA learns a look and its movement from single clips. A Control LoRA learns
                the change between a pair, so every item needs a reference clip as well as a target.
                Upstream calls that an IC-LoRA.
              </span>
            </label>
          )}

          <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
            Max clip length (seconds)
            <NumberField
              label=""
              value={hp.clipSeconds ?? 1}
              step={0.5}
              onChange={(v) => set('clipSeconds', v)}
            />
            {/* The grid snaps DOWN, so asking for 5s trains on 4.458s. Silent until it was shown
                here, and a user lost the action at the end of their clips to it. */}
            <span className="text-[10px] text-zinc-500">
              Trains on{' '}
              <span className="text-zinc-300">
                {resolved?.frames} frames ({resolved?.seconds.toFixed(2)}s)
              </span>{' '}
              per clip. This VAE encodes only {clipGrid.grid}n+{clipGrid.offset} frames at{' '}
              {clipGrid.fps}fps, so the length snaps down, never up.
            </span>
            <span className="text-[10px] text-zinc-600">
              Stills ignore this, and the floor is{' '}
              {((clipGrid.grid + clipGrid.offset) / clipGrid.fps).toFixed(2)}s. Length costs time
              rather than memory: a clip peaks about the same as a still, but a second of video is
              roughly four times the step time.
            </span>
          </label>

          <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
            Clip window
            <select
              value={hp.clipWindow ?? 'start'}
              onChange={(e) => set('clipWindow', e.target.value as 'start' | 'end')}
              className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
            >
              <option value="start">From the start</option>
              <option value="end">From the end</option>
            </select>
            <span className="text-[10px] text-zinc-600">
              Snapping down always drops part of each clip. Pick the end when the action you want to
              teach is at the finish.
            </span>
          </label>
        </>
      )}

      {QUANTIZABLE.includes(arch) && (
        <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
          Base precision
          <select
            value={hp.baseQuant ?? 'auto'}
            onChange={(e) => set('baseQuant', e.target.value as TrainingBaseQuant)}
            className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
          >
            {quantsFor(arch).map((q) => (
              <option key={q.value} value={q.value}>
                {q.label}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-zinc-600">
            {arch === 'moss-image'
              ? 'int8 freezes the base at 5.4GB and trains the LoRA full precision: about 6.5GB at 512px instead of 11GB.'
              : '4-bit freezes the base at NF4 and trains the LoRA full precision: ~12GB at 512px instead of ~30GB.'}
          </span>
        </label>
      )}

      {QUANTIZABLE.includes(arch) && (hp.baseQuant ?? 'auto') !== 'nf4' && (
        <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
          CPU offload
          <select
            value={hp.offload ?? 'auto'}
            onChange={(e) => set('offload', e.target.value as TrainingOffload)}
            className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
          >
            {OFFLOADS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-zinc-600">
            Streams activations to system RAM so a bf16 base fits a smaller card. Needed for bf16
            1024 on ~45GB cards; slower, but keeps full precision.
          </span>
        </label>
      )}

      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Rank" value={hp.rank} onChange={(v) => set('rank', v)} />
        <NumberField label="Alpha" value={hp.alpha} onChange={(v) => set('alpha', v)} />
        <NumberField
          label="Learning rate"
          value={hp.learningRate}
          step={0.00001}
          onChange={(v) => set('learningRate', v)}
        />
        <NumberField label="Steps" value={hp.steps} step={100} onChange={(v) => set('steps', v)} />
        <NumberField
          label="Batch size"
          value={hp.batchSize}
          onChange={(v) => set('batchSize', v)}
        />
        <NumberField
          label="Resolution"
          value={hp.resolution}
          step={64}
          onChange={(v) => set('resolution', v)}
        />
        <NumberField
          label="Save every"
          value={hp.saveEvery}
          step={50}
          onChange={(v) => set('saveEvery', v)}
        />
      </div>

      <label className="flex items-start gap-2 text-[11px] text-zinc-400">
        <input
          type="checkbox"
          checked={hp.saveSnapshots ?? false}
          onChange={(e) => set('saveSnapshots', e.target.checked)}
          className="mt-0.5 accent-emerald-500"
        />
        <span className="flex flex-col gap-0.5">
          <span className="text-zinc-200">Keep a LoRA at every checkpoint</span>
          <span className="text-zinc-500">
            Saves a usable LoRA at each save point, so you can pick the step before a run
            overtrains. A stopped run always keeps one, whatever this is set to.
          </span>
        </span>
      </label>

      <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
        LoRA scope
        <select
          value={hp.loraScope ?? 'full'}
          onChange={(e) => set('loraScope', e.target.value as TrainingLoraScope)}
          className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
        >
          {SCOPES.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <span className="text-[10px] text-zinc-600">
          Narrow to attention for long runs, where adapting everything costs prompt adherence.
        </span>
      </label>

      <div className="grid grid-cols-2 gap-3">
        <NumberField
          label="Caption dropout"
          value={hp.captionDropout ?? 0}
          step={0.05}
          onChange={(v) => set('captionDropout', Math.min(1, Math.max(0, v)))}
        />
        <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
          Flip images
          <button
            onClick={() => set('flipAugment', !hp.flipAugment)}
            className={`rounded-md border px-2 py-1 text-left text-sm ${
              hp.flipAugment
                ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300'
                : 'border-border bg-black/30 text-zinc-400 hover:bg-panel'
            }`}
          >
            {hp.flipAugment ? 'On' : 'Off'}
          </button>
        </label>
      </div>
      <span className="-mt-1 text-[10px] text-zinc-600">
        Dropout trains some steps without the caption so the LoRA holds without the trigger. Flip
        mirrors every image, doubling the dataset - wrong for text or anything asymmetric.
      </span>

      {gpus.length > 1 && (
        <div className="flex flex-col gap-1 text-[11px] text-zinc-400">
          GPUs (leave empty for auto)
          <div className="flex flex-wrap gap-2">
            {gpus.map((g) => (
              <button
                key={g.index}
                onClick={() => toggleGpu(g.index)}
                className={`rounded-md border px-2 py-1 text-[11px] ${
                  hp.gpuIds.includes(g.index)
                    ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300'
                    : 'border-border text-zinc-400 hover:bg-panel'
                }`}
              >
                GPU {g.index}
              </button>
            ))}
          </div>
        </div>
      )}
      <Modal
        open={confirming}
        onClose={() => setConfirming(false)}
        title="Discard checkpoints and start a new session?"
      >
        <div className="flex flex-col gap-4 p-5">
          <p className="text-sm text-zinc-300">
            This run has a checkpoint you can currently resume. That checkpoint was built with the
            previous rank, LoRA targets and base model, so it cannot be continued under the new
            settings.
          </p>
          <p className="text-sm text-zinc-400">
            Applying will delete its checkpoints and cached dataset, and the node will offer Start
            instead of Resume. The LoRA files any finished run already produced are not touched.
          </p>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setConfirming(false)}
              className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-300 hover:bg-panel"
            >
              Keep checkpoints
            </button>
            <button
              onClick={() => void commit()}
              className="rounded-md bg-red-500/90 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-500"
            >
              Discard and apply
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}

/** Caption node settings. Small today, but it is where the Adjust button has to lead: a button that
 * silently toggled a hidden flag read as broken, because nothing on the node showed what it did. */
function CaptionSettings({
  itemId,
  overwrite,
}: {
  itemId: string
  overwrite: boolean
}): React.JSX.Element {
  const { patchData, toggleSettings } = useBoardActions()
  return (
    <div className="absolute right-0 top-0 z-40 flex h-full w-80 flex-col gap-3 overflow-y-auto border-l border-border bg-panel p-4 shadow-2xl">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-200">Caption settings</span>
        <button
          onClick={() => toggleSettings(itemId)}
          className="rounded p-1 text-zinc-400 hover:bg-panel hover:text-zinc-200"
          title="Close"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </div>
      <label className="flex items-start gap-2 text-[11px] text-zinc-400">
        <input
          type="checkbox"
          checked={overwrite}
          onChange={(e) => void patchData(itemId, { overwrite: e.target.checked })}
          className="mt-0.5 accent-emerald-500"
        />
        <span className="flex flex-col gap-0.5">
          <span className="text-zinc-200">Re-caption every image</span>
          <span className="text-zinc-500">
            Off, only images with an empty caption are captioned, so hand-written captions survive a
            re-run. On, every caption is replaced.
          </span>
        </span>
      </label>
    </div>
  )
}
