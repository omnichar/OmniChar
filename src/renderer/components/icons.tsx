/**
 * Material Design icons as inline SVGs. We can't load the Material Symbols web
 * font (the renderer CSP only allows `'self'` for styles/scripts and no external
 * font-src), so these mirror the official 24×24 Material glyph paths and follow
 * the codebase's existing inline-SVG icon convention.
 *
 * Each icon takes a `className` (default sizes to 1em via `h-[1em] w-[1em]`) and
 * paints with `currentColor` so callers control colour/size through text utils.
 */

function Svg({
  className,
  children,
  // Newer Material Symbols glyphs are drawn on a 960 grid with a flipped origin, so an icon
  // copied from those has to bring its own box rather than be rescaled by hand.
  viewBox = '0 0 24 24',
}: {
  className?: string
  children: React.ReactNode
  viewBox?: string
}): React.JSX.Element {
  return (
    <svg
      viewBox={viewBox}
      fill="currentColor"
      className={className ?? 'h-[1em] w-[1em]'}
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

/** `image` - pictures / assets. */
export function ImageIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
    </Svg>
  )
}

/** `face` - a saved character (not the Control Space rig, which is a posable figure). */
export function CharacterIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM9 8c.83 0 1.5.67 1.5 1.5S9.83 11 9 11s-1.5-.67-1.5-1.5S8.17 8 9 8zm3 10c-2.28 0-4.22-1.66-5-4h10c-.78 2.34-2.72 4-5 4zm3-7c-.83 0-1.5-.67-1.5-1.5S14.17 8 15 8s1.5.67 1.5 1.5S15.83 11 15 11z" />
    </Svg>
  )
}

/** `history` - timeline / past takes. */
export function HistoryIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
    </Svg>
  )
}

/** `rebase_edit` - the published workflow catalogue. */
export function WorkflowIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className} viewBox="0 -960 960 960">
      <path d="M400-80v-164l294-292q12-12 26.5-18t30.5-6q16 0 30.5 6t26.5 18l49 50q11 12 17 26.5t6 29.5q0 15-6.5 29.5T856-374L564-80H400Zm397-348-50-49 50 49ZM480-160h50l162-162-25-25-25-25-162 162v50Zm187-187-25-25 25 25 25 25-25-25ZM430-590l-56-57 73-73H313q-9 26-28 45t-45 28v334q35 13 57.5 43.5T320-200q0 50-35 85t-85 35q-50 0-85-35t-35-85q0-39 22.5-69t57.5-43v-335q-35-13-57.5-43.5T80-760q0-50 35-85t85-35q39 0 69.5 22.5T313-800h134l-73-73 56-57 170 170-170 170Zm415-255q35 35 35 85t-35 85q-35 35-85 35t-85-35q-35-35-35-85t35-85q35-35 85-35t85 35ZM200-160q17 0 28.5-11.5T240-200q0-17-11.5-28.5T200-240q-17 0-28.5 11.5T160-200q0 17 11.5 28.5T200-160Zm0-560q17 0 28.5-11.5T240-760q0-17-11.5-28.5T200-800q-17 0-28.5 11.5T160-760q0 17 11.5 28.5T200-720Zm560 0q17 0 28.5-11.5T800-760q0-17-11.5-28.5T760-800q-17 0-28.5 11.5T720-760q0 17 11.5 28.5T760-720ZM200-200Zm0-560Zm560 0Z" />
    </Svg>
  )
}

/** `place_item` - a workflow imported onto the canvas. Deliberately not a download arrow: the
 *  card shows both counts and they must not read as the same number. */
export function ImportIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className} viewBox="0 -960 960 960">
      <path d="M200-120q-33 0-56.5-23.5T120-200v-400q0-33 23.5-56.5T200-680h160v80H200v400h560v-400H600v-80h160q33 0 56.5 23.5T840-600v400q0 33-23.5 56.5T760-120H200Zm280-200L320-480l56-56 64 63v-487h80v487l64-63 56 56-160 160Z" />
    </Svg>
  )
}

/** `visibility` - a view count. */
export function EyeIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 6.5c3.79 0 7.17 2.13 8.82 5.5-1.65 3.37-5.03 5.5-8.82 5.5S4.83 15.37 3.18 12C4.83 8.63 8.21 6.5 12 6.5m0-2C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5m0 5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5m0-2a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9" />
    </Svg>
  )
}

/** `open_in_new` - a link that leaves the app. */
export function ExternalLinkIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3z" />
    </Svg>
  )
}

/** `edit` - pencil. */
export function EditIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
    </Svg>
  )
}

/** `delete` - a waste basket, for destructive actions that are not "close". */
export function TrashIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
    </Svg>
  )
}

/** `create_new_folder` - folder with a +. */
export function CreateNewFolderIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-1 8h-3v3h-2v-3h-3v-2h3V9h2v3h3v2z" />
    </Svg>
  )
}

/** `file_download` - arrow into a tray, used for import. */
export function DownloadIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" />
    </Svg>
  )
}

/** `add` - plus sign. */
export function PlusIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
    </Svg>
  )
}

/** `folder` - a standard directory. */
export function FolderIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" />
    </Svg>
  )
}

/** `settings` - a gear. */
/** Outlined "i", carrying a setting's explanation so the panel stays a list of controls. */
export function InfoIcon({
  className,
  title,
}: {
  className?: string
  title?: string
}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className ?? 'h-[1em] w-[1em]'}
      // Titled: it is the setting's explanation, so it must reach a screen reader too.
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
    >
      {title ? <title>{title}</title> : null}
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  )
}

export function SettingsIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" />
    </Svg>
  )
}

/** Two-sparkle mark - flags AI-generated outputs (mirrors the Generate node's badge). */
export function SparklesIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 3l1.9 4.8L18.6 9.7 13.9 11.6 12 16.4 10.1 11.6 5.4 9.7 10.1 7.8Z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z" />
    </Svg>
  )
}

/** `star` (filled) - the chosen hero take. */
export function StarIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
    </Svg>
  )
}

/** `music_note` - an audio file. */
export function MusicNoteIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
    </Svg>
  )
}

/** `mic` - a character's voice. */
export function MicIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z" />
    </Svg>
  )
}

/** `close` - an X. */
export function CloseIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
    </Svg>
  )
}

/** `content_copy` - copy to clipboard. */
export function CopyIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" />
    </Svg>
  )
}

/** `check` - a tick, e.g. "copied". */
export function CheckIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
    </Svg>
  )
}

/** `chevron_right` - a right-pointing caret. */
export function ChevronRightIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z" />
    </Svg>
  )
}

/** `chevron_left` - a left-pointing caret. */
export function ChevronLeftIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M15.41 16.59 10.83 12l4.58-4.59L14 6l-6 6 6 6z" />
    </Svg>
  )
}

/** `expand_more` - a down-pointing caret. */
export function ChevronDownIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z" />
    </Svg>
  )
}

/** `model_training` - LoRA training. */
export function TrainIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M5 3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h4v2H7v2h10v-2h-2v-2h4a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5zm0 2h14v10H5V5zm7 1.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm0 2a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" />
    </Svg>
  )
}

/** `dashboard` / grid - a training dataset (a stack of images). */
export function DatasetIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M3 3h8v8H3V3zm2 2v4h4V5H5zm8-2h8v8h-8V3zm2 2v4h4V5h-4zM3 13h8v8H3v-8zm2 2v4h4v-4H5zm8-2h8v8h-8v-8zm2 2v4h4v-4h-4z" />
    </Svg>
  )
}

/** `inventory_2` - the Trainer sidebar's Outputs tab (produced LoRAs + resumable runs). */
export function LoraOutputIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M20 2H4a2 2 0 0 0-2 2v3h20V4a2 2 0 0 0-2-2zM2 9v11a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9H2zm7 3h6v2H9v-2z" />
    </Svg>
  )
}

/** `dashboard_customize` - the Studio (node canvas) tab. */
export function StudioIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M3 3h8v6H3V3zm0 8h8v10H3V11zm10 6h8v4h-8v-4zm0-14h8v10h-8V3z" />
    </Svg>
  )
}

/** `play_arrow` - run / start. */
export function PlayIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M8 5v14l11-7z" />
    </Svg>
  )
}

/** `deployed_code` - a model weight file / the Models panel. */
export function ModelsIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M12 2 3 7v10l9 5 9-5V7l-9-5zm0 2.3 6.5 3.6L12 11.5 5.5 7.9 12 4.3zM5 9.6l6 3.3v6.5l-6-3.3V9.6zm8 9.8v-6.5l6-3.3v6.5l-6 3.3z" />
    </Svg>
  )
}

/** `movie` - video material. Distinct from `TrainIcon`, which is the training glyph. */
export function MovieIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V4h-4z" />
    </Svg>
  )
}

/** `refresh` - re-scan / reload. */
export function RefreshIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Svg className={className}>
      <path d="M17.65 6.35A8 8 0 1 0 19.73 14h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
    </Svg>
  )
}
