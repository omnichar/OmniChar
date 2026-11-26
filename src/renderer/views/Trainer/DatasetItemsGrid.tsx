/** The dataset panel: what a dataset holds, with one entry point for changing it. */
import { useMemo, useState } from 'react'
import { pairCaptions } from '@shared/captionPairing'
import { readCaptionFiles } from '@/lib/captionFiles'
import { uploadFiles } from '@/lib/importFiles'
import { useAssetStore } from '../../store/assetStore'
import { useTrainingStore } from '../../store/trainingStore'
import { AddDataDialog } from './AddDataDialog'
import { PairTile } from './PairTile'
import { ipcErrorMessage } from '../../lib/ipcError'

// Clips are accepted for the archs that can train on them (MiniMax H3). An arch that cannot is
// handed images only by the precache, so a stray clip is skipped rather than breaking a run.
const isMedia = (f: File): boolean =>
  f.type.startsWith('image/') ||
  f.type.startsWith('video/') ||
  /\.(png|jpe?g|webp|bmp|mp4|mov|webm|mkv|avi)$/i.test(f.name)

export function DatasetItemsGrid({ datasetId }: { datasetId: string }): React.JSX.Element {
  // `?? []` outside the selector: returning a fresh [] from the selector loops the store (Object.is).
  const items = useTrainingStore((s) => s.itemsByDataset[datasetId]) ?? []
  const addItems = useTrainingStore((s) => s.addItems)
  const addFromPath = useTrainingStore((s) => s.addFromPath)
  const setCaption = useTrainingStore((s) => s.setCaption)
  const removeItem = useTrainingStore((s) => s.removeItem)
  const assets = useAssetStore((s) => s.assets)
  const loadAssets = useAssetStore((s) => s.load)
  const setError = useTrainingStore((s) => s.setError)
  const [busy, setBusy] = useState(false)
  const [fromPath, setFromPath] = useState(false)
  const [adding, setAdding] = useState(false)
  const [path, setPath] = useState('')
  const [pathError, setPathError] = useState<string | null>(null)
  // Highlight while OS files are dragged over the grid.
  const [fileOver, setFileOver] = useState(false)

  const byId = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])

  /**
   * Import dropped/picked files into the Library and attach them to this dataset. A `.txt` with the
   * same basename as an image (`1.png` + `1.txt`) is read as that image's caption, matching the
   * ComfyUI/kohya dataset convention. Auto-caption then skips anything that already has a caption.
   */
  const loadPath = async (): Promise<void> => {
    const target = path.trim()
    if (!target) return
    setBusy(true)
    setPathError(null)
    try {
      const failure = await addFromPath(datasetId, target)
      if (failure) setPathError(failure)
      else {
        setPath('')
        setFromPath(false)
      }
    } finally {
      setBusy(false)
    }
  }

  const attachFiles = async (files: File[]): Promise<void> => {
    setBusy(true)
    try {
      const media = files.filter(isMedia)
      const captions = await readCaptionFiles(files)
      // Captions dropped on their own: pair them to items already in the dataset by filename.
      if (!media.length) {
        const named = items.flatMap((it) => {
          const asset = byId.get(it.assetId)
          return asset ? [{ item: it, name: asset.name }] : []
        })
        const paired = pairCaptions(
          named.map((n) => n.name),
          captions,
        )
        await Promise.all(
          named.map(({ item, name }) => {
            const caption = paired.get(name)
            return caption ? setCaption(datasetId, item.id, caption) : undefined
          }),
        )
        return
      }
      const uploaded = await uploadFiles(media, null)
      if (!uploaded.length) return
      const created = await addItems(
        datasetId,
        uploaded.map((a) => a.id),
      )
      // Pair each new item to its sidecar caption by the image's original filename.
      const itemByAsset = new Map(created.map((it) => [it.assetId, it]))
      const paired = pairCaptions(
        uploaded.map((a) => a.name),
        captions,
      )
      await Promise.all(
        uploaded.map((asset) => {
          const caption = paired.get(asset.name)
          const item = itemByAsset.get(asset.id)
          return caption && item ? setCaption(datasetId, item.id, caption) : undefined
        }),
      )
      await loadAssets() // so thumbnails resolve for the freshly uploaded assets
    } catch (e) {
      setError(ipcErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  // True when a drag carries OS files (Finder/Explorer), not an internal asset drag.
  const isFileDrag = (e: React.DragEvent): boolean =>
    Array.from(e.dataTransfer.types).includes('Files')

  const onDragOver = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!fileOver) setFileOver(true)
  }

  const onDrop = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    setFileOver(false)
    // Keep images and their .txt captions; readCaptionFiles/attachFiles sort them out.
    const files = Array.from(e.dataTransfer.files ?? []).filter(
      (f) => isMedia(f) || f.name.toLowerCase().endsWith('.txt'),
    )
    if (files.length > 0) void attachFiles(files)
  }

  return (
    <div
      className={`relative flex min-h-0 flex-1 flex-col gap-3 rounded-lg ${
        fileOver ? 'ring-2 ring-inset ring-accent' : ''
      }`}
      onDragOver={onDragOver}
      onDragLeave={(e) => {
        // Only clear when the cursor actually leaves the grid, not a child element.
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFileOver(false)
      }}
      onDrop={onDrop}
    >
      {fileOver && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-accent/5">
          <span className="rounded-md border border-accent bg-panel/90 px-3 py-1.5 text-xs text-accent">
            Drop images or clips (and .txt captions) to add to this dataset
          </span>
        </div>
      )}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setAdding(true)}
          disabled={busy}
          className="rounded-md border border-accent/40 px-3 py-1.5 text-xs font-medium text-accent hover:bg-accent/10 disabled:opacity-40"
        >
          {busy ? 'Adding…' : 'Add/Manage Training Data'}
        </button>

        <span className="text-[11px] text-zinc-500">{items.length} items</span>
      </div>

      {fromPath && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <input
              value={path}
              onChange={(e) => {
                setPath(e.target.value)
                setPathError(null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void loadPath()
              }}
              placeholder="/path/to/a/folder of images or clips"
              spellCheck={false}
              className="flex-1 rounded-md border border-border bg-black/30 px-2 py-1.5 text-sm text-zinc-100 outline-none focus:border-zinc-500"
            />
            <button
              onClick={() => void loadPath()}
              disabled={busy || !path.trim()}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-zinc-200 hover:bg-panel disabled:opacity-40"
            >
              {busy ? 'Importing…' : 'Import'}
            </button>
          </div>
          <span className={`text-[10px] ${pathError ? 'text-red-400' : 'text-zinc-600'}`}>
            {pathError ??
              'A folder on the machine running Inline Studio, not your browser. Copies every image and clip in it, and reads any matching .txt as the caption.'}
          </span>
        </div>
      )}

      {items.length === 0 ? (
        <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border px-6 text-center text-sm text-zinc-500">
          Drop images here, or use Add files. A .txt next to an image (1.png + 1.txt) is read as its
          caption. 10 to 30 of your character trains a good LoRA.
        </div>
      ) : (
        // content-start: the grid fills the panel, so without it one row of cards stretches to
        // the bottom instead of sizing to the thumbnail.
        <div className="grid min-h-0 flex-1 auto-rows-min content-start grid-cols-2 gap-3 overflow-y-auto pr-1 sm:grid-cols-3 lg:grid-cols-4">
          {items.map((item) => {
            const asset = byId.get(item.assetId)
            return (
              <div
                key={item.id}
                className="group flex flex-col rounded-md border border-border bg-surface/60"
              >
                <div className="relative">
                  <PairTile
                    target={asset}
                    reference={item.referenceAssetId ? byId.get(item.referenceAssetId) : undefined}
                    className="rounded-t-md"
                  />
                  <button
                    onClick={() => void removeItem(datasetId, item.id)}
                    className="absolute right-1 top-1 hidden rounded bg-black/70 px-1.5 py-0.5 text-[11px] text-white group-hover:block"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {adding && (
        <AddDataDialog datasetId={datasetId} assets={assets} onClose={() => setAdding(false)} />
      )}
    </div>
  )
}
