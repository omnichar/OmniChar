/**
 * State for LoRA training: datasets, their items, training runs, and the live run telemetry.
 * Mutations go through `studio().training.*`; live progress/samples/host-stats arrive via the
 * training events (subscribed by `subscribeTrainingEvents`, wired in App) which call the
 * setters here. Mirrors `generationStore`'s shape.
 */
import { create } from 'zustand'
import type {
  SystemStatsEvent,
  CaptionerModel,
  DatasetRepoPreview,
  StagedDatasetItem,
  TrainingDataset,
  TrainingMode,
  TrainingDatasetItem,
  TrainingDoneEvent,
  TrainingErrorEvent,
  TrainingHyperparams,
  TrainingProgressEvent,
  TrainingRun,
  CaptionProgressEvent,
  TrainingLogEvent,
  TrainingSampleEvent,
  TrainingSnapshot,
} from '@shared/types'
import { pairCaptions } from '@shared/captionPairing'
import { isCaptionFile, readCaptionFiles } from '../lib/captionFiles'
import { uploadFiles } from '../lib/importFiles'
import { useAssetStore } from './assetStore'
import { studio } from '@/lib/studio'
import { useMoodboardStore } from './moodboardStore'
import { ipcErrorMessage } from '../lib/ipcError'

/** Live progress for one run, updated from `onTrainingProgress`. */
export interface RunProgress {
  fraction: number
  step: number
  totalSteps: number
  status?: string
}

interface TrainingState {
  datasets: TrainingDataset[]
  itemsByDataset: Record<string, TrainingDatasetItem[]>
  runs: TrainingRun[]
  activeDatasetId: string | null
  captioning: boolean
  captioners: CaptionerModel[]
  error: string | null

  progressByRun: Record<string, RunProgress>
  lossByRun: Record<string, number[]>
  /** The trainer subprocess's streamed stdout, per run - shown in the Trainer node. */
  logsByRun: Record<string, string[]>
  /** Live auto-caption progress per dataset, cleared when the captioner exits. */
  captionProgress: Record<string, { done: number; total: number }>
  samplesByRun: Record<string, string[]>
  /** Mid-run LoRAs per run, oldest first. */
  snapshotsByRun: Record<string, TrainingSnapshot[]>
  systemStats: SystemStatsEvent | null

  loadDatasets: () => Promise<void>
  createDataset: (name: string, triggerWord: string) => Promise<TrainingDataset | null>
  /** Rename a dataset or change its trigger word; the trigger lands on the next run's captions. */
  updateDataset: (datasetId: string, name: string, triggerWord: string) => Promise<void>
  selectDataset: (datasetId: string | null) => void
  loadItems: (datasetId: string) => Promise<void>
  /** Returns the created items, so a caller can pair captions to the assets it just added. */
  addItems: (datasetId: string, assetIds: string[]) => Promise<TrainingDatasetItem[]>
  /** Import a folder on the Core machine, sidecar captions included. Returns an error string. */
  addFromPath: (datasetId: string, path: string) => Promise<string | null>
  removeItem: (datasetId: string, itemId: string) => Promise<void>
  /** Remove every image from a dataset in one go, refetching once when done. */
  removeAll: (datasetId: string) => Promise<void>
  setCaption: (datasetId: string, itemId: string, caption: string) => Promise<void>
  /** Pair a reference clip with an item, or clear it with null. Control LoRAs only. */
  setItemReference: (
    datasetId: string,
    itemId: string,
    referenceAssetId: string | null,
  ) => Promise<void>
  /** Switch a dataset between clip and control training. */
  setDatasetMode: (datasetId: string, mode: TrainingMode) => Promise<void>
  /** Summarise a Hugging Face dataset repo without downloading it. */
  inspectDatasetPath: (path: string) => Promise<DatasetRepoPreview | null>
  inspectDatasetRepo: (repo: string) => Promise<DatasetRepoPreview | null>
  /** Stage a folder: assets are imported, but no dataset row is written. */
  /** Upload picked files and pair them, without touching any dataset. */
  stageFiles: (files: File[]) => Promise<StagedDatasetItem[] | null>
  /** Caption staged assets directly; returns asset id -> caption. */
  captionAssets: (assetIds: string[], model?: string) => Promise<Record<string, string>>
  stageFromPath: (path: string) => Promise<StagedDatasetItem[] | null>
  /** The same for a Hugging Face dataset, downloading it first. */
  stageFromRepo: (repo: string) => Promise<StagedDatasetItem[] | null>
  /** Accept staged rows into a dataset. */
  commitStaged: (datasetId: string, rows: StagedDatasetItem[]) => Promise<void>
  autoCaption: (datasetId: string, overwrite: boolean, model?: string) => Promise<void>
  /** The caption models Core offers; loaded once, lazily. */
  loadCaptioners: () => Promise<void>
  loadRuns: () => Promise<void>
  /** Returns the created run so a canvas node can persist its `runId` and rebind after a reload. */
  start: (datasetId: string, hyperparams: TrainingHyperparams) => Promise<TrainingRun | null>
  resume: (runId: string) => Promise<void>
  cancel: (runId: string) => Promise<void>
  /** Drop a run's checkpoints so a changed configuration starts clean. */
  discard: (runId: string) => Promise<void>

  applyProgress: (e: TrainingProgressEvent) => void
  applySample: (e: TrainingSampleEvent) => void
  applySnapshot: (e: TrainingSnapshot) => void
  loadSnapshots: (runId: string) => Promise<void>
  /** Copy a snapshot into models/loras/ so a Load LoRA node can select it. */
  exportSnapshot: (runId: string, step: number) => Promise<string | null>
  applyLog: (e: TrainingLogEvent) => void
  applyCaptionProgress: (e: CaptionProgressEvent) => void
  applyDone: (e: TrainingDoneEvent) => void
  applyError: (e: TrainingErrorEvent) => void
  applyStats: (e: SystemStatsEvent) => void
  setError: (error: string | null) => void
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  datasets: [],
  itemsByDataset: {},
  runs: [],
  activeDatasetId: null,
  captioning: false,
  captioners: [],
  error: null,
  progressByRun: {},
  lossByRun: {},
  samplesByRun: {},
  snapshotsByRun: {},
  logsByRun: {},
  captionProgress: {},
  systemStats: null,

  loadDatasets: async () => {
    const res = await studio().training.listDatasets()
    if (res.ok) set({ datasets: res.value })
    else set({ error: res.error })
  },

  createDataset: async (name, triggerWord) => {
    const res = await studio().training.createDataset({ name, triggerWord })
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    set((s) => ({ datasets: [res.value, ...s.datasets], activeDatasetId: res.value.id }))
    return res.value
  },

  updateDataset: async (datasetId, name, triggerWord) => {
    const res = await studio().training.updateDataset(datasetId, { name, triggerWord })
    if (!res.ok) return set({ error: res.error })
    set((s) => ({ datasets: s.datasets.map((d) => (d.id === datasetId ? res.value : d)) }))
  },

  selectDataset: (activeDatasetId) => {
    set({ activeDatasetId })
    if (activeDatasetId) void get().loadItems(activeDatasetId)
  },

  loadItems: async (datasetId) => {
    const res = await studio().training.listItems(datasetId)
    if (res.ok) set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
    else set({ error: res.error })
  },

  addItems: async (datasetId, assetIds) => {
    const res = await studio().training.addItems(datasetId, assetIds)
    if (!res.ok) {
      set({ error: res.error })
      return []
    }
    await get().loadItems(datasetId)
    // Pairing can fold a reference into another item, so the mode may have changed too.
    await get().loadDatasets()
    // The import creates library assets too; without reloading them the new items have no asset to
    // resolve and every tile renders empty.
    await useAssetStore.getState().load()
    return res.value
  },

  addFromPath: async (datasetId, path) => {
    const res = await studio().training.addFromPath(datasetId, path)
    if (!res.ok) {
      // Returned rather than only stored: a bad path is the common case here and the field wants
      // to show it inline, next to what the reader typed.
      set({ error: res.error })
      return ipcErrorMessage(res.error)
    }
    await get().loadItems(datasetId)
    // The import creates library assets too; without reloading them the new items have no asset to
    // resolve and every tile renders empty.
    await useAssetStore.getState().load()
    return null
  },

  removeItem: async (datasetId, itemId) => {
    const res = await studio().training.removeItem(itemId)
    if (!res.ok) return set({ error: res.error })
    await get().loadItems(datasetId)
  },

  removeAll: async (datasetId) => {
    const ids = (get().itemsByDataset[datasetId] ?? []).map((it) => it.id)
    const results = await Promise.all(ids.map((id) => studio().training.removeItem(id)))
    const failed = results.find((r) => !r.ok)
    if (failed && !failed.ok) set({ error: failed.error })
    await get().loadItems(datasetId)
  },

  setCaption: async (datasetId, itemId, caption) => {
    const res = await studio().training.setCaption(itemId, caption)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({
      itemsByDataset: {
        ...s.itemsByDataset,
        [datasetId]: (s.itemsByDataset[datasetId] ?? []).map((it) =>
          it.id === itemId ? res.value : it,
        ),
      },
    }))
  },

  setItemReference: async (datasetId, itemId, referenceAssetId) => {
    const res = await studio().training.setItemReference(itemId, referenceAssetId)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({
      itemsByDataset: {
        ...s.itemsByDataset,
        [datasetId]: (s.itemsByDataset[datasetId] ?? []).map((it) =>
          it.id === itemId ? res.value : it,
        ),
      },
    }))
  },

  setDatasetMode: async (datasetId, mode) => {
    const res = await studio().training.setDatasetMode(datasetId, mode)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({
      datasets: s.datasets.map((d) => (d.id === datasetId ? res.value : d)),
    }))
  },

  inspectDatasetPath: async (path) => {
    const res = await studio().training.inspectDatasetPath(path)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    return res.value
  },

  inspectDatasetRepo: async (repo) => {
    const res = await studio().training.inspectDatasetRepo(repo)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    return res.value
  },

  captionAssets: async (assetIds, model) => {
    set({ captioning: true, error: null })
    try {
      const res = await studio().training.captionAssets(assetIds, model)
      if (!res.ok) {
        set({ error: res.error })
        return {}
      }
      return res.value
    } finally {
      set({ captioning: false })
    }
  },

  stageFiles: async (files) => {
    // Captions are read here, not uploaded: Core imports media only, so a .txt never reached it.
    const captions = await readCaptionFiles(files)
    const assets = await uploadFiles(
      files.filter((f) => !isCaptionFile(f)),
      null,
    )
    if (!assets.length) return []
    const res = await studio().training.stageAssets(assets.map((a) => a.id))
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    await useAssetStore.getState().load()
    const paired = pairCaptions(
      res.value.map((row) => row.name),
      captions,
    )
    return res.value.map((row) => ({ ...row, caption: paired.get(row.name) ?? row.caption }))
  },

  stageFromPath: async (path) => {
    const res = await studio().training.stageFromPath(path)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    // Assets exist now even though no dataset row does, so previews can resolve.
    await useAssetStore.getState().load()
    return res.value
  },

  stageFromRepo: async (repo) => {
    const res = await studio().training.stageFromRepo(repo)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    await useAssetStore.getState().load()
    return res.value
  },

  commitStaged: async (datasetId, rows) => {
    const res = await studio().training.commitStaged(datasetId, rows)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
    await get().loadDatasets()
  },

  autoCaption: async (datasetId, overwrite, model) => {
    set({ captioning: true, error: null })
    try {
      const res = await studio().training.autoCaption(datasetId, overwrite, model)
      if (res.ok) set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
      else set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    } finally {
      set({ captioning: false })
    }
  },

  loadCaptioners: async () => {
    if (get().captioners.length) return
    const res = await studio().training.captioners()
    if (res.ok) set({ captioners: res.value })
  },

  loadRuns: async () => {
    const res = await studio().training.listRuns()
    if (res.ok) set({ runs: res.value })
    else set({ error: res.error })
  },

  start: async (datasetId, hyperparams) => {
    const res = await studio().training.start(datasetId, hyperparams)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    set((s) => ({ runs: [res.value, ...s.runs.filter((r) => r.id !== res.value.id)] }))
    return res.value
  },

  resume: async (runId) => {
    const res = await studio().training.resume(runId)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({ runs: s.runs.map((r) => (r.id === runId ? res.value : r)) }))
  },

  discard: async (runId) => {
    const res = await studio().training.discard(runId)
    if (!res.ok) return
    set((s) => ({ runs: s.runs.map((r) => (r.id === runId ? res.value : r)) }))
  },
  cancel: async (runId) => {
    const res = await studio().training.cancel(runId)
    if (!res.ok) set({ error: res.error })
  },

  applyProgress: (e) =>
    set((s) => ({
      progressByRun: {
        ...s.progressByRun,
        [e.runId]: {
          fraction: e.fraction,
          step: e.step,
          totalSteps: e.totalSteps,
          status: e.status,
        },
      },
      lossByRun:
        typeof e.loss === 'number'
          ? { ...s.lossByRun, [e.runId]: [...(s.lossByRun[e.runId] ?? []), e.loss] }
          : s.lossByRun,
    })),

  applySample: (e) =>
    set((s) => ({
      samplesByRun: { ...s.samplesByRun, [e.runId]: [...(s.samplesByRun[e.runId] ?? []), e.path] },
    })),

  applySnapshot: (e) =>
    set((s) => {
      const existing = s.snapshotsByRun[e.runId] ?? []
      // The trainer can re-emit a step on resume, so key on the step rather than appending.
      const next = [...existing.filter((x) => x.step !== e.step), e].sort((a, b) => a.step - b.step)
      return { snapshotsByRun: { ...s.snapshotsByRun, [e.runId]: next } }
    }),

  loadSnapshots: async (runId) => {
    try {
      const res = await studio().training.snapshots(runId)
      if (res.ok) {
        set((s) => ({ snapshotsByRun: { ...s.snapshotsByRun, [runId]: res.value } }))
      }
    } catch {
      // A backend that predates the channel simply has no snapshots to list.
    }
  },

  exportSnapshot: async (runId, step) => {
    try {
      const res = await studio().training.exportSnapshot(runId, step)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      return res.value.path
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  applyLog: (e) =>
    set((s) => ({
      // Capped so a long run can't grow the buffer without bound; the node shows the tail. Deep
      // enough to still hold the setup and precache lines once a few hundred steps have streamed
      // in, since those are what a "Copy logs" for a failed run needs to include.
      logsByRun: {
        ...s.logsByRun,
        [e.runId]: [...(s.logsByRun[e.runId] ?? []), e.line].slice(-3000),
      },
    })),

  applyCaptionProgress: (e) =>
    set((s) => {
      const next = { ...s.captionProgress }
      if (e.finished) delete next[e.datasetId]
      else next[e.datasetId] = { done: e.done, total: e.total }
      return { captionProgress: next }
    }),

  applyDone: (e) => {
    set((s) => ({
      runs: s.runs.map((r) =>
        r.id === e.runId
          ? { ...r, status: 'done', outputLoraPath: e.outputLoraPath, progressFraction: 1 }
          : r,
      ),
    }))
    void get().loadRuns()
  },

  applyError: (e) => {
    set((s) => ({
      runs: s.runs.map((r) => (r.id === e.runId ? { ...r, error: e.error } : r)),
    }))
    // A run ending (cancel/crash) only broadcasts the error - reload so its new status lands,
    // otherwise a stopped run still reads as "training" and its Stop control never flips back.
    void get().loadRuns()
  },

  applyStats: (systemStats) => set({ systemStats }),
  setError: (error) => set({ error }),
}))

/** Wire the training + telemetry events to the store. Returns an unsubscribe fn. */
export function subscribeTrainingEvents(): () => void {
  const s = useTrainingStore.getState()
  const unsubs = [
    studio().events.onTrainingProgress((e) => s.applyProgress(e)),
    studio().events.onTrainingSample((e) => s.applySample(e)),
    studio().events.onTrainingSnapshot((e) => s.applySnapshot(e)),
    studio().events.onTrainingLog((e) => s.applyLog(e)),
    studio().events.onCaptionProgress((e) => s.applyCaptionProgress(e)),
    studio().events.onTrainingDone((e) => s.applyDone(e)),
    // A graph-run Train LoRA node learns its run id here; the board holds it so Resume survives.
    studio().events.onTrainingNodeBound((e) => {
      void useMoodboardStore.getState().patchItemData(e.itemId, { runId: e.runId })
    }),
    studio().events.onTrainingError((e) => s.applyError(e)),
    studio().events.onSystemStats((e) => s.applyStats(e)),
  ]
  return () => unsubs.forEach((u) => u())
}
