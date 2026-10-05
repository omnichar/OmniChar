/**
 * The declarative contract for a generation node (a fal.ai model surfaced on the canvas).
 *
 * A `NodeDef` is pure data + pure functions - no Electron, no fs, no network. The renderer
 * uses it to render input ports + param widgets; the main-process executor uses it to build
 * the fal request and parse the response. Adding a model = one new file + a registry line.
 *
 * This file is imported by BOTH the renderer and main, so it must stay shell-agnostic
 * (open-core rule - see MEMORY: open-core-packages).
 */

/** The common input/output type system. Ports declare what kind of media flows through them. */
export type PortKind =
  | 'image'
  | 'image[]'
  | 'video'
  | 'video[]'
  | 'audio'
  | 'audio[]'
  | 'text'
  | 'path'
  // A saved identity wired in from a Core `character/load` node; it carries a filename, not media.
  | 'character'

/** A typed input socket on a node (left side). */
export interface InputPort {
  id: string
  label: string
  kind: PortKind
  /** If required and unwired, the node can't run (the play button stays disabled). */
  required: boolean
}

/** A typed output socket on a node (right side). */
export interface OutputPort {
  id: string
  label: string
  kind: PortKind
}

/**
 * One user-editable parameter, rendered generically by GenNode as a widget.
 * `advanced: true` hides it behind the node's "More options" disclosure (only primary params
 * - those without the flag - show by default).
 */
export type ParamField =
  | { key: string; label: string; widget: 'text' | 'textarea'; default: string; advanced?: boolean }
  | {
      key: string
      label: string
      widget: 'select'
      options: { value: string; label: string }[]
      default: string
      advanced?: boolean
    }
  | {
      key: string
      label: string
      widget: 'number'
      default: number
      min?: number
      max?: number
      step?: number
      advanced?: boolean
    }
  | { key: string; label: string; widget: 'boolean'; default: boolean; advanced?: boolean }

/** Concrete param values for a node instance (persisted on its backing Frame). */
export type ParamValues = Record<string, string | number | boolean>

/**
 * A node's inputs after the executor has resolved + uploaded them to fal storage. Grouped by
 * kind so `buildRequest` can map them to the model's specific fields (e.g. `image_url` vs
 * `image_urls`). Empty arrays when nothing is wired to that kind of port.
 */
export interface ResolvedInputs {
  images: string[]
  masks: string[]
  videos: string[]
  audios: string[]
  texts: string[]
  /**
   * The same URIs keyed by the input port each was wired to. Only defs with two ports of one kind
   * need this - read it through `portMedia`, never directly, so untagged inputs still resolve.
   */
  byHandle: Record<string, string[]>
  /** A wired character, already normalised for this endpoint by Core. Absent when none is wired. */
  character?: AppliedCharacter
}

/** What a reference is of. Mirrors `charfile.ROLES`; the split is decided when a character is built. */
export type CharacterRole = 'face' | 'body' | 'cloth'

export const CHARACTER_ROLES: readonly CharacterRole[] = ['face', 'body', 'cloth']

/** Only the form an endpoint documents resolves: ordinal prose, H3's `<Picture N>`, Seedance's `@ImageN`. */
export type CharacterPromptStyle = 'ordinal' | 'token' | 'at-image'

/** Composed by Core, so ref ordering, the role split and the numbering stay where they already live. */
export interface AppliedCharacter {
  name: string
  /** Normalised references as data URIs, in the order they must be sent. */
  refs: string[]
  /** One role (`face` | `body` | `cloth`) per ref, in the same order. */
  roles: string[]
  /** Prepended to the user's prompt; names the positions the refs land on. */
  promptPrefix: string
  /** The character's voice as a data URI, when the endpoint asked for it and the character has one. */
  voice?: string
}

/** Params alone cannot price a model that bills per reference image. */
export interface PriceInputs {
  /** Images that will reach the model's reference port, the character's own included. */
  referenceImages?: number
}

/** An estimated cost for one generation. Models price differently (per image / MP / second / …). */
export interface PriceEstimate {
  /** Estimated total USD for one run with the current params. */
  amount: number
  /** Always an estimate (fal prices can change; some models bill on real output size). */
  approx?: boolean
}

/**
 * Compact USD label for a price estimate, e.g. `~$0.145`. Sub-dollar costs use 3 decimals so
 * models that differ only in the third place (e.g. $0.145 vs $0.136) read distinctly; a dollar or
 * more rounds to cents.
 */
export function formatPrice(est: PriceEstimate): string {
  const s = `$${est.amount < 1 ? est.amount.toFixed(3) : est.amount.toFixed(2)}`
  return est.approx ? `~${s}` : s
}

/**
 * A generation node definition. Its functions carry ALL model-specific knowledge and are
 * pure/unit-testable; the executor stays model-agnostic.
 *
 * Note there is deliberately no `parseOutputs` here: the browser builds the request
 * (`resolveEndpoint` + `buildRequest`) and hands it to Core, which then submits, polls, and
 * downloads the result asynchronously - the browser is never in the loop for the response. Parsing
 * therefore lives in Core (`inline_core/studio/fal.py: parse_outputs`), keyed on `outputKind`.
 */
export interface NodeDef {
  /** The fal model id, e.g. `openai/gpt-image-2`. Also the registry key + `Frame.modelId`. */
  id: string
  /** Display title, e.g. `GPT Image 2`. */
  title: string
  /** Grouping label for the node palette, e.g. `Image` / `Video`. */
  category: string
  provider: 'fal'
  /** The kind of media this node produces → the backing `Frame.kind`. */
  outputKind: 'image' | 'video' | 'audio'
  /**
   * When true, the node runs without a connected Prompt node - the model derives its own prompt
   * from its media inputs (e.g. Sonilo reads the video), and a wired prompt only steers the result.
   * Default (absent) keeps the prompt required, which is the norm.
   */
  promptOptional?: boolean
  inputs: InputPort[]
  params: ParamField[]
  outputs: OutputPort[]
  /** How a wired character applies here, or absent on a node that takes none. */
  character?: {
    port: string
    style: CharacterPromptStyle
    maxImages: number
    maxRefs: number
    excludeRoles?: CharacterRole[]
    /** Name each role in the prompt. Needed once identity references are gone, or the text claims
     *  a photo of an outfit shows the character's face. */
    roleLines?: boolean
    /** The audio port a character's voice joins, after the wired clips. Absent takes no voice. */
    voicePort?: string
  }
  /** Pick the fal endpoint from the resolved inputs (e.g. text-to-image vs image-to-image). PURE. */
  resolveEndpoint(resolved: ResolvedInputs): string
  /**
   * Build the fal request body from the frame's persisted param values + resolved (uploaded) input
   * URLs. Params arrive as `unknown` (JSON from the DB), so implementations coerce defensively. PURE.
   */
  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown>
  /** Estimate the USD cost of one generation from the current param values. PURE; optional. */
  estimatePrice?(params: Record<string, unknown>, wired?: PriceInputs): PriceEstimate | null
}

/** Fill a param-value map from a node def's field defaults (used when creating a fal frame). */
export function defaultParams(def: NodeDef): ParamValues {
  const out: ParamValues = {}
  for (const field of def.params) {
    out[field.key] = field.default
  }
  return out
}

/**
 * The media wired to one of a def's input ports.
 *
 * Prefers the explicit per-port routing (`byHandle`, set when the user wires an edge into a specific
 * dot). Falls back to the kind buckets for untagged inputs - drag-drop, and every input made before
 * inputs recorded their port - handing the def's Nth port of that kind the Nth untagged item, or the
 * whole bucket for a list port. For a def with one port of a kind this is exactly the old behaviour.
 */
export function portMedia(def: NodeDef, resolved: ResolvedInputs, portId: string): string[] {
  const explicit = resolved.byHandle[portId]
  if (explicit?.length) return explicit
  const port = def.inputs.find((p) => p.id === portId)
  if (!port) return []
  const family = mediaFamily(port.kind)
  if (!family) return []
  const bucket =
    family === 'video' ? resolved.videos : family === 'audio' ? resolved.audios : resolved.images
  // Anything already claimed by an explicit wire is not up for positional fallback.
  const claimed = new Set(Object.values(resolved.byHandle).flat())
  const untagged = bucket.filter((uri) => !claimed.has(uri))
  if (isListPort(port.kind)) return untagged
  const peers = def.inputs.filter((p) => mediaFamily(p.kind) === family)
  const picked = untagged[peers.indexOf(port)]
  return picked === undefined ? [] : [picked]
}

/** Which resolved-input bucket a port draws from, or null for non-media kinds. */
export function mediaFamily(kind: PortKind): 'image' | 'video' | 'audio' | null {
  if (kind === 'image' || kind === 'image[]') return 'image'
  if (kind === 'video' || kind === 'video[]') return 'video'
  if (kind === 'audio' || kind === 'audio[]') return 'audio'
  return null
}

/** Appended, never led: the numbers on the node face come from the wires, and must keep meaning them. */
export function withCharacterRefs(
  def: NodeDef,
  resolved: ResolvedInputs,
  portId: string,
): string[] {
  const wired = portMedia(def, resolved, portId)
  if (def.character?.port !== portId || !resolved.character) return wired
  return [...wired, ...resolved.character.refs].slice(0, def.character.maxImages)
}

/** Appended after the wired clips, whose `<Audio N>` numbers the user's prompt already names. */
export function withCharacterVoice(
  def: NodeDef,
  resolved: ResolvedInputs,
  portId: string,
): string[] {
  const wired = portMedia(def, resolved, portId)
  const voice = resolved.character?.voice
  if (def.character?.voicePort !== portId || !voice) return wired
  return [...wired, voice]
}

/** The user's prompt with a wired character's binding text in front of it. */
export function withCharacterPrompt(resolved: ResolvedInputs, prompt: string): string {
  return resolved.character ? resolved.character.promptPrefix + prompt : prompt
}

/** True when a port accepts several wires, so its order carries meaning. */
export function isListPort(kind: PortKind): boolean {
  return kind === 'image[]' || kind === 'video[]' || kind === 'audio[]'
}

/** An empty `ResolvedInputs` (all kinds empty) - a convenience for callers/tests. */
export function emptyResolvedInputs(): ResolvedInputs {
  return { images: [], masks: [], videos: [], audios: [], texts: [], byHandle: {} }
}
