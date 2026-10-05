/**
 * The published model list, and which files something asked for are absent.
 *
 * A missing file is reported whether or not the registry can supply it: the name and the folder are
 * what let someone place their own file, and only a match adds a download.
 */
import { create } from 'zustand'
import type { MissingModel, RegistryModel } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

/** Files the user already closed the popup on; it only reopens for one they have not seen. */
const DISMISSED_KEY = 'inline.missingModels.dismissed'

function readDismissed(): Set<string> {
  try {
    const raw = localStorage.getItem(DISMISSED_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return new Set(
      Array.isArray(parsed) ? parsed.filter((p): p is string => typeof p === 'string') : [],
    )
  } catch {
    return new Set()
  }
}

function writeDismissed(paths: Set<string>): void {
  try {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify([...paths]))
  } catch {
    // Private mode or blocked storage: the popup simply stays quiet for this session only.
  }
}

/** Opens only when a missing file has not been dismissed before. Pure, so it is testable. */
export function hasUndismissed(missing: MissingModel[], dismissed: Set<string>): boolean {
  return missing.some((row) => !dismissed.has(row.path))
}

export interface ModelRequest {
  filename: string
  category?: string
}

interface ModelRegistryState {
  entries: RegistryModel[]
  /** True when the registry was unreachable and these came from the on-disk cache. */
  stale: boolean
  loading: boolean
  error: string | null
  /** The open popup's missing files, or null when it is closed. */
  missing: MissingModel[] | null
  /** What opened the popup, so the user knows why they are seeing it. */
  reason: string
  /** Paths of missing files the user has closed the popup on. */
  dismissed: Set<string>
  downloading: Record<string, { fraction: number; status: string }>

  load: (refresh?: boolean) => Promise<void>
  check: (wanted: ModelRequest[], reason: string) => Promise<number>
  dismiss: () => void
  download: (modelId: string) => Promise<void>
  /** Queue every missing file that the registry can actually supply. */
  downloadAll: () => Promise<void>
  cancel: (modelId: string) => Promise<void>
}

export const useModelRegistryStore = create<ModelRegistryState>((set) => ({
  entries: [],
  stale: false,
  loading: false,
  error: null,
  missing: null,
  reason: '',
  dismissed: readDismissed(),
  downloading: {},

  load: async (refresh = false) => {
    set({ loading: true, error: null })
    try {
      const res = await studio().models.registry(refresh)
      if (!res.ok) return set({ loading: false, error: res.error })
      set({ entries: res.value.entries, stale: res.value.stale, loading: false })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  check: async (wanted, reason) => {
    if (wanted.length === 0) return 0
    try {
      const res = await studio().models.resolveMissing(wanted)
      if (!res.ok) return 0
      const missing = res.value.missing
      if (hasUndismissed(missing, useModelRegistryStore.getState().dismissed)) {
        set({ missing, reason })
      }
      return missing.length
    } catch {
      // A registry that cannot be reached must never block placing a node.
      return 0
    }
  },

  dismiss: () =>
    set((s) => {
      const dismissed = new Set(s.dismissed)
      for (const row of s.missing ?? []) dismissed.add(row.path)
      writeDismissed(dismissed)
      return { missing: null, reason: '', dismissed }
    }),

  download: async (modelId) => {
    set((s) => ({
      downloading: { ...s.downloading, [modelId]: { fraction: 0, status: 'Starting…' } },
    }))
    try {
      const res = await studio().models.downloadRegistry(modelId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  downloadAll: async () => {
    // Only what the registry can supply and does not already have: a file with no match has nowhere
    // to be fetched from, and queueing it would leave a row that never moves.
    const { missing, download, downloading } = useModelRegistryStore.getState()
    for (const row of missing ?? []) {
      // The same match the row's own button would fetch, or Download all quietly picks a different
      // precision than the one beside it.
      const match = preferredMatch(row)
      if (!match || match.present || downloading[match.model.id]) continue
      await download(match.model.id)
    }
  },

  cancel: async (modelId) => {
    // Cleared here rather than on the event: a queued one has no run to report anything.
    set((s) => ({ downloading: without(s.downloading, modelId) }))
    try {
      await studio().models.cancelRegistryDownload(modelId)
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))

/** The registry may carry the same file at several precisions; an exact filename match wins. */
export function preferredMatch(row: MissingModel): MissingModel['matches'][number] | null {
  return row.matches.find((m) => m.exact) ?? row.matches[0] ?? null
}

/** A row whose matched model just landed now reports it present, so it stops offering Download. */
export function markPresent(rows: MissingModel[] | null, modelId: string): MissingModel[] | null {
  if (!rows) return rows
  return rows.map((row) =>
    row.matches.some((m) => m.model.id === modelId && !m.present)
      ? {
          ...row,
          matches: row.matches.map((m) => (m.model.id === modelId ? { ...m, present: true } : m)),
        }
      : row,
  )
}

function without<T>(map: Record<string, T>, key: string): Record<string, T> {
  const next = { ...map }
  delete next[key]
  return next
}

/** Model downloads report against the node type they were started for; registry ones use this. */
const REGISTRY_NODE = 'registry'

export function subscribeModelRegistry(): () => void {
  const events = studio().events
  const onProgress = events.onModelDownloadProgress((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    useModelRegistryStore.setState((s) => ({
      downloading: {
        ...s.downloading,
        [e.componentId]: { fraction: e.fraction, status: e.status ?? '' },
      },
    }))
  })
  const onDone = events.onModelDownloadDone((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    // The popup's rows are the snapshot taken when it opened, so clearing the progress alone left
    // the row offering Download again - the one action that is certainly wrong once it has landed.
    useModelRegistryStore.setState((s) => ({
      downloading: without(s.downloading, e.componentId),
      missing: markPresent(s.missing, e.componentId),
    }))
    void useModelRegistryStore.getState().load()
  })
  const onError = events.onModelDownloadError((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    useModelRegistryStore.setState((s) => ({
      downloading: without(s.downloading, e.componentId),
      error: e.error,
    }))
  })
  return () => {
    onProgress()
    onDone()
    onError()
  }
}
