/**
 * The "missing models" popup for a Core model node (opened from the node's blinking hint). Lists the
 * model components the node needs, whether each is on disk, and where it lives - with a Download
 * button per component and a "Download all". **Nothing is downloaded automatically**: this is the one
 * explicit place a model is fetched, and it writes into `core/models/` (never a hidden cache).
 */
import { useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import type { ModelComponent, ModelFitEstimate } from '@shared/coreNodes'
import { DownloadIcon, XIcon } from './NodeBadge'

function ComponentRow({
  cacheKey,
  comp,
}: {
  /** Where this popup's answer is cached, which is the node type unless the caller keyed it. */
  cacheKey: string
  comp: ModelComponent
}): React.JSX.Element {
  const download = useModelRequirementsStore((s) => s.download)
  const asked = useModelRequirementsStore((s) => s.asked[cacheKey])
  const dl = useModelRequirementsStore((s) => s.downloads[cacheKey]?.[comp.id])
  // A present component always reads as "Ready" - never as an in-progress bar (a sibling's reload can
  // land while this one is mid-download; presence wins).
  const busy = dl !== undefined && dl.error === undefined && !comp.present
  const pct = Math.round((dl?.fraction ?? 0) * 100)

  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-border bg-surface/60 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-[12px] font-medium text-zinc-100">{comp.label}</span>
          <span className="truncate font-mono text-[10px] text-zinc-500">{comp.source}</span>
        </span>
        {comp.present ? (
          <span className="shrink-0 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-medium text-accent">
            Ready
          </span>
        ) : comp.optional && !busy ? (
          <button
            onClick={() => void download(cacheKey, comp.id, asked?.nodeType, asked?.params)}
            title={`Optional. ${comp.label}. Not required to generate.`}
            className="flex shrink-0 items-center gap-1 rounded-md border border-border bg-panel px-2 py-1 text-[10px] font-medium text-zinc-300 hover:border-emerald-500/50 hover:text-emerald-300"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            Suggested
          </button>
        ) : busy ? (
          <span className="shrink-0 text-[10px] font-medium text-emerald-300">{pct}%</span>
        ) : (
          <button
            onClick={() => void download(cacheKey, comp.id, asked?.nodeType, asked?.params)}
            className="flex shrink-0 items-center gap-1 rounded-md border border-border bg-panel px-2 py-1 text-[10px] font-medium text-zinc-100 hover:border-emerald-500/50 hover:text-emerald-300"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            Download
          </button>
        )}
      </div>

      {busy && (
        <div className="h-1 overflow-hidden rounded-full bg-black/40">
          <div className="h-full bg-emerald-400 transition-all" style={{ width: `${pct}%` }} />
        </div>
      )}
      {busy && dl?.status && <span className="text-[10px] text-zinc-500">{dl.status}</span>}
      {dl?.error && <span className="text-[10px] text-red-400">Download failed: {dl.error}</span>}
    </div>
  )
}

const formatGb = (mb: number | null): string => (mb === null ? '?' : `${Math.round(mb / 1024)}GB`)

// Caller gates this on allPresent: the estimate sizes the files on disk, so a partial set reads as fitting.
function FitNote({ estimate }: { estimate: ModelFitEstimate }): React.JSX.Element | null {
  if (!estimate.warning) return null
  const tone = estimate.fits
    ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
    : 'border-rose-500/40 bg-rose-500/10 text-rose-200'
  return (
    <div className={`flex flex-col gap-1 border-b border-border px-4 py-2.5 text-[11px] ${tone}`}>
      <span className="font-medium">{estimate.warning}</span>
      <span className="font-mono text-[10px] opacity-80">
        needs {formatGb(estimate.requiredVramMb)} &middot; card {formatGb(estimate.totalVramMb)}{' '}
        &middot; RAM free {formatGb(estimate.freeRamMb)}
      </span>
    </div>
  )
}

export function ModelRequirementsModal(): React.JSX.Element | null {
  const cacheKey = useModelRequirementsStore((s) => s.openFor)
  const close = useModelRequirementsStore((s) => s.close)
  const download = useModelRequirementsStore((s) => s.download)
  const asked = useModelRequirementsStore((s) => (cacheKey ? s.asked[cacheKey] : undefined))
  const reqs = useModelRequirementsStore((s) => (cacheKey ? s.byType[cacheKey] : undefined))
  const anyBusy = useModelRequirementsStore((s) =>
    cacheKey
      ? Object.values(s.downloads[cacheKey] ?? {}).some((d) => d.error === undefined)
      : false,
  )
  // The descriptor is keyed by node type, and a keyed entry's cache key is not one.
  const nodeType = asked?.nodeType ?? cacheKey ?? ''
  const title = useCoreNodesStore(
    (s) => s.descriptors.find((d) => d.type === nodeType)?.title ?? nodeType,
  )

  if (!cacheKey || !reqs) return null
  // Optional (suggested) components never count as "missing" or block "Download all".
  const missing = reqs.components.filter((c) => !c.present && !c.optional)

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={close}
    >
      <div
        className="flex max-h-full w-[28rem] max-w-[90vw] flex-col overflow-hidden rounded-xl border border-border bg-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
          <div className="flex min-w-0 flex-col">
            <span className="truncate text-sm font-semibold text-zinc-100">{title} models</span>
            <span className="text-[10px] text-zinc-500">
              {reqs.allPresent ? 'All models are on disk.' : `${missing.length} missing`}
            </span>
          </div>
          <button
            onClick={close}
            aria-label="Close"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-surface hover:text-zinc-100"
          >
            <XIcon className="h-4 w-4" />
          </button>
        </header>

        {reqs.allPresent && reqs.estimate ? <FitNote estimate={reqs.estimate} /> : null}

        <p className="border-b border-border px-4 py-2 text-[11px] leading-relaxed text-zinc-500">
          These files must be on disk to generate. Nothing is downloaded automatically - grab what's
          missing below and it's saved into{' '}
          <span className="font-mono text-zinc-400">core/models/</span>.
        </p>

        <div className="flex flex-col gap-2 overflow-y-auto p-4">
          {reqs.components.map((comp) => (
            <ComponentRow key={comp.id} cacheKey={cacheKey} comp={comp} />
          ))}
        </div>

        <footer className="flex items-center justify-between gap-2 border-t border-border px-4 py-3">
          <button
            onClick={() => void download(cacheKey, 'all', nodeType, asked?.params)}
            disabled={reqs.allPresent || anyBusy}
            className="flex items-center gap-1.5 rounded-md bg-emerald-500/90 px-3 py-1.5 text-[11px] font-semibold text-black hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <DownloadIcon className="h-4 w-4" />
            Download all
          </button>
          <button
            onClick={close}
            className="rounded-md border border-border px-3 py-1.5 text-[11px] font-medium text-zinc-300 hover:bg-surface hover:text-zinc-100"
          >
            Done
          </button>
        </footer>
      </div>
    </div>
  )
}
