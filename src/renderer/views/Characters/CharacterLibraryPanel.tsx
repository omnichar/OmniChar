import { useEffect, useState } from 'react'
import type { CharacterSummary } from '@shared/types'
import { useAssetStore } from '../../store/assetStore'
import { useCharacterStore } from '../../store/characterStore'
import { getAssetDragIds } from '../../lib/dnd'
import { CloseIcon, DownloadIcon, EditIcon, MicIcon, PlusIcon } from '../../components/icons'
import { buildCharacterEditChain, buildCharacterStarter } from '../../lib/characterChains'
import { useUiStore } from '../../store/uiStore'

/** Where a chain lands: what the user is looking at. */
const centre = (): { x: number; y: number } => useUiStore.getState().canvasCenter

/** Every saved character, plus the two ways to make one: the drop-in form, or the canvas chain. */
export function CharacterLibraryPanel(): React.JSX.Element {
  const { characters, loading, error, load, remove, importFile } = useCharacterStore()
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    void load()
  }, [load])

  const acceptsDrop = (e: React.DragEvent): boolean =>
    Array.from(e.dataTransfer.types).some((t) => t === 'Files' || t.startsWith('application/x-'))

  const onDragOver = (e: React.DragEvent): void => {
    if (!acceptsDrop(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!dragOver) setDragOver(true)
  }

  const onDrop = async (e: React.DragEvent): Promise<void> => {
    if (!acceptsDrop(e)) return
    e.preventDefault()
    setDragOver(false)

    // Assets dragged from the library become a new character's references.
    const assetIds = getAssetDragIds(e.dataTransfer)
    if (assetIds.length > 0) return void buildCharacterStarter(centre(), assetIds)

    const files = Array.from(e.dataTransfer.files ?? [])
    const chars = files.filter((f) => f.name.toLowerCase().endsWith('.char'))
    for (const file of chars) await importFile(file)

    const images = files.filter((f) => f.type.startsWith('image/'))
    if (images.length > 0) {
      // Route new images through the asset library first, so a character's references are also
      // ordinary assets the user can find, re-use and delete.
      await useAssetStore.getState().importFiles(images)
      const added = useAssetStore.getState().assets.slice(0, images.length)
      await buildCharacterStarter(
        centre(),
        added.map((a) => a.id),
      )
    }
  }

  return (
    <div
      className={`flex h-full flex-col ${dragOver ? 'ring-2 ring-inset ring-accent' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false)
      }}
      onDrop={(e) => void onDrop(e)}
    >
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-xs font-medium text-muted">Characters</span>
        <button
          type="button"
          title="Build a character on the canvas"
          onClick={() => void buildCharacterStarter(centre())}
          className="flex h-6 w-6 items-center justify-center rounded text-muted hover:bg-surface hover:text-fg"
        >
          <PlusIcon className="h-4 w-4" />
        </button>
      </div>

      {error && <div className="px-2 py-1 text-[11px] text-red-400">{error}</div>}

      <div className="flex-1 space-y-2 overflow-y-auto p-2">
        {characters.map((character) => (
          <CharacterCard
            key={character.file}
            character={character}
            onOpen={() => void buildCharacterEditChain(character.file, centre())}
            onDelete={() => void remove(character.file)}
          />
        ))}
        {!loading && characters.length === 0 && (
          <div className="rounded border border-dashed border-border p-4 text-center text-[11px] text-muted">
            Drop an image here to make a character. One is enough.
          </div>
        )}
      </div>
    </div>
  )
}

function CharacterCard({
  character,
  onOpen,
  onDelete,
}: {
  character: CharacterSummary
  onOpen: () => void
  onDelete: () => void
}): React.JSX.Element {
  // The X sits under the cursor on hover, and deleting is irreversible, so it asks first.
  const [confirming, setConfirming] = useState(false)
  if (character.error) {
    return (
      <div className="rounded border border-red-500/40 bg-red-500/5 p-2 text-[11px] text-red-300">
        <div className="font-medium">{character.name}</div>
        <div className="mt-0.5 text-red-400/80">{character.error}</div>
      </div>
    )
  }
  return (
    <div className="group relative rounded border border-border bg-surface/60 hover:border-zinc-600">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-center gap-2 p-2 text-left"
      >
        <img
          src={`/character-ref/${encodeURIComponent(character.file)}/0`}
          alt=""
          className="h-10 w-10 shrink-0 rounded object-cover"
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs text-fg">{character.name}</span>
          <span className="flex items-center gap-1 truncate text-[10px] text-muted">
            {character.refs} reference{character.refs === 1 ? '' : 's'}
            {character.voiceSeconds != null && (
              <span
                title={`Voice, ${Math.round(character.voiceSeconds)}s. Applied on MiniMax H3 Reference to Video only.`}
                className="flex items-center gap-0.5 text-violet-300"
              >
                <MicIcon className="h-3 w-3" />
                voice
              </span>
            )}
          </span>
        </span>
      </button>
      <div className="absolute right-1 top-1 hidden gap-0.5 group-hover:flex">
        <button
          type="button"
          title="Edit this character"
          onClick={(e) => {
            e.stopPropagation()
            onOpen()
          }}
          className="flex h-5 w-5 items-center justify-center rounded bg-black/60 text-muted hover:text-fg"
        >
          <EditIcon className="h-3 w-3" />
        </button>
        <a
          href={`/download/character/${encodeURIComponent(character.file)}`}
          download={character.file}
          title="Export this character"
          onClick={(e) => e.stopPropagation()}
          className="flex h-5 w-5 items-center justify-center rounded bg-black/60 text-muted hover:text-fg"
        >
          <DownloadIcon className="h-3 w-3" />
        </a>
        <button
          type="button"
          title="Delete this character"
          onClick={(e) => {
            e.stopPropagation()
            setConfirming(true)
          }}
          className="flex h-5 w-5 items-center justify-center rounded bg-black/60 text-muted hover:text-red-400"
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      </div>
      {confirming && (
        <div className="absolute inset-0 flex items-center justify-center gap-1 rounded bg-black/85 px-2">
          <span className="truncate text-[10px] text-muted">Delete?</span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            className="rounded bg-red-600 px-1.5 py-0.5 text-[10px] text-white hover:bg-red-500"
          >
            Delete
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setConfirming(false)
            }}
            className="rounded px-1.5 py-0.5 text-[10px] text-muted hover:text-fg"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}
