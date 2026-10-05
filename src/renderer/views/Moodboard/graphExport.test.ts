import { beforeEach, describe, expect, it } from 'vitest'
import type { InlineStudioApi } from '@shared/ipc'
import { ok } from '@shared/result'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { parseRecipeJson } from '../../lib/pngRecipe'
import { setStudioClient } from '../../lib/studio'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useModelRequirementsStore } from '../../store/modelRequirementsStore'
import { useModelRegistryStore } from '../../store/modelRegistryStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import type { NodeDescriptor } from '@shared/coreNodes'
import { duplicateGraph, graphJson, graphRecipe, graphSlice, unsupportedTypes } from './graphExport'

function item(id: string, type: string, data: Record<string, unknown> = {}): MoodboardItem {
  return {
    id,
    projectId: 'p',
    surface: 'studio',
    type,
    assetId: null,
    frameId: null,
    parentId: null,
    data,
    x: 0,
    y: 0,
    width: 100,
    height: 100,
    rotation: 0,
    zIndex: 0,
    createdAt: 0,
    updatedAt: 0,
  } as MoodboardItem
}

function connector(from: string, to: string): MoodboardConnector {
  return {
    id: `${from}-${to}`,
    projectId: 'p',
    surface: 'studio',
    fromItemId: from,
    toItemId: to,
    label: null,
    data: { sourceHandle: 'out', targetHandle: 'prompt' },
    createdAt: 0,
  } as MoodboardConnector
}

const PROMPT = item('prompt1', 'prompt', { promptText: 'a neon city' })
const CORE = item('core1', 'core', {
  core: { type: 'alibaba/z-image-turbo', params: { steps: 8 }, output: { takeId: 'tk1' } },
})
const STRAY = item('core2', 'core', { core: { type: 'other' } })

beforeEach(() => {
  useMoodboardStore.setState({
    items: [PROMPT, CORE, STRAY],
    connectors: [connector('prompt1', 'core1')],
  })
  // The export reads the descriptor for each param's kind, and Core for what the node requires
  // without naming it.
  useCoreNodesStore.setState({
    descriptors: [
      {
        type: 'alibaba/z-image-turbo',
        title: 'Z-Image Turbo',
        category: 'Generate',
        icon: 'wand',
        outputKind: 'image',
        inputs: [],
        outputs: [],
        params: [{ key: 'steps', label: 'Steps', widget: 'number', default: 8, kind: 'number' }],
      },
    ] as unknown as NodeDescriptor[],
  })
  useModelRegistryStore.setState({ entries: [] })
  useModelRequirementsStore.setState({ byType: {} })
  setStudioClient({
    models: {
      requirements: async (nodeType: string) =>
        ok({
          allPresent: nodeType !== 'alibaba/z-image-turbo',
          components:
            nodeType === 'alibaba/z-image-turbo'
              ? [
                  {
                    id: 'te',
                    label: 'Text encoder',
                    category: 'text_encoders',
                    present: false,
                    localPath: 'text_encoders/qwen_3_4b.safetensors',
                    repo: 'org/repo',
                    source: 'org/repo/qwen_3_4b.safetensors',
                    optional: false,
                  },
                ]
              : [],
        }),
    },
  } as unknown as InlineStudioApi)
})

describe('graphSlice', () => {
  it('takes only the connected graph, not the whole board', () => {
    const slice = graphSlice('core1')
    expect(slice.items.map((i) => i.id).sort()).toEqual(['core1', 'prompt1'])
    expect(slice.connectors).toHaveLength(1)
  })

  it('an unconnected node is its own one-node graph', () => {
    expect(graphSlice('core2').items.map((i) => i.id)).toEqual(['core2'])
    expect(graphSlice('core2').connectors).toEqual([])
  })
})

describe('graphRecipe', () => {
  it('emits the recipe shape the PNG importer already reads', async () => {
    const recipe = await graphRecipe('core1')
    expect(recipe.app).toBe('inline-studio')
    expect(recipe.version).toBe(2)
    expect(recipe.target).toBe('core1')
    expect(recipe.coreType).toBe('alibaba/z-image-turbo')
    expect(recipe.params).toEqual({ steps: 8 })
    expect(recipe.prompt).toBe('a neon city')
    expect(recipe.graph?.items).toHaveLength(2)
    expect(recipe.graph?.connectors).toEqual([
      {
        fromItemId: 'prompt1',
        toItemId: 'core1',
        data: { sourceHandle: 'out', targetHandle: 'prompt' },
      },
    ])
  })

  it('keeps the take history out of the exported graph', async () => {
    // A recipe describes how to make the image, not the images already made.
    const core = (await graphRecipe('core1')).graph?.items.find((i) => i.id === 'core1')
    expect((core?.data.core as Record<string, unknown>).output).toBeUndefined()
  })

  it('round-trips through JSON unchanged', async () => {
    const recipe = await graphRecipe('core1')
    expect(JSON.parse(JSON.stringify(recipe))).toEqual(recipe)
  })

  it('exported JSON is accepted by the drop-side parser', async () => {
    // Export then drop is the whole point: the file has to satisfy the same `app` guard the PNG
    // chunk does, or dragging it back onto the canvas silently does nothing.
    const parsed = parseRecipeJson(await graphJson('core1'))
    expect(parsed).not.toBeNull()
    expect(parsed?.target).toBe('core1')
    expect(parsed?.graph?.items).toHaveLength(2)
    expect(parsed?.graph?.connectors).toHaveLength(1)
  })
})

describe('an unpicked character', () => {
  it('exports empty instead of naming the first character in the catalog', async () => {
    useMoodboardStore.setState({
      items: [item('load1', 'core', { core: { type: 'character/load', params: { file: '' } } })],
      connectors: [],
    })
    useCoreNodesStore.setState({
      descriptors: [
        {
          type: 'character/load',
          title: 'Load Character',
          category: 'Character',
          icon: 'sparkles',
          inputs: [],
          outputs: [],
          params: [
            {
              key: 'file',
              label: 'Character',
              widget: 'select',
              default: '',
              optionsFrom: 'characters',
              options: [{ value: 'got.char', label: 'got.char' }],
            },
          ],
        },
      ] as unknown as NodeDescriptor[],
    })
    const recipe = await graphRecipe('load1')
    expect(recipe.params).toEqual({ file: '' })
  })
})

describe('unsupportedTypes', () => {
  it('is empty for a graph the importer can rebuild', () => {
    expect(unsupportedTypes('core1')).toEqual([])
  })

  it('names the node types a re-import would drop', () => {
    const director = item('dir1', 'director')
    useMoodboardStore.setState({
      items: [CORE, director],
      connectors: [connector('core1', 'dir1')],
    })
    expect(unsupportedTypes('core1')).toEqual(['director'])
  })

  it('is empty for a training graph, so Train LoRA can offer the same menu', () => {
    // The Trainer was the one node whose Run control carried no graph menu, on the grounds that
    // copying a training job made no sense. Every `train/*` type is rebuildable, so it did.
    const dataset = item('ds1', 'train/dataset', { datasetId: 'd1' })
    const caption = item('cap1', 'train/caption')
    const trainer = item('tr1', 'train/lora')
    useMoodboardStore.setState({
      items: [dataset, caption, trainer],
      connectors: [connector('ds1', 'cap1'), connector('cap1', 'tr1')],
    })
    expect(unsupportedTypes('tr1')).toEqual([])
    expect(graphSlice('tr1').items.map((i) => i.type)).toEqual([
      'train/dataset',
      'train/caption',
      'train/lora',
    ])
  })
})

describe('duplicateGraph', () => {
  let created: MoodboardItem[]
  let connectorCalls: { from: string; to: string; sh: string | null; th: string | null }[]

  beforeEach(() => {
    created = []
    connectorCalls = []
    let n = 0
    const make = (type: string, x: number, y: number): MoodboardItem => {
      n += 1
      const copy = { ...item(`new${n}`, type), x, y }
      created.push(copy)
      return copy
    }
    setStudioClient({
      moodboard: {
        addCoreNode: async (_type: string, x: number, y: number) => ok(make('core', x, y)),
        addPrompt: async (x: number, y: number) => ok(make('prompt', x, y)),
        updateItem: async (id: string, patch: Record<string, unknown>) => {
          const target = created.find((c) => c.id === id)
          return ok({ ...(target as MoodboardItem), ...patch })
        },
        createConnector: async (from: string, to: string, sh: string | null, th: string | null) => {
          connectorCalls.push({ from, to, sh, th })
          return ok({ ...connector(from, to), id: `c${connectorCalls.length}` })
        },
      },
    } as unknown as InlineStudioApi)
  })

  it('copies a core + prompt graph and re-creates its wiring', async () => {
    // copyOne used to fall through to `default: return null` for core/prompt, so this whole
    // action silently did nothing on a generation graph.
    const count = await duplicateGraph('core1')

    expect(count).toBe(2)
    expect(created.map((c) => c.type).sort()).toEqual(['core', 'prompt'])
    expect(connectorCalls).toHaveLength(1)
    expect(connectorCalls[0].sh).toBe('out')
    expect(connectorCalls[0].th).toBe('prompt')
    // Wired between the copies, not back to the originals.
    expect(connectorCalls[0].from.startsWith('new')).toBe(true)
    expect(connectorCalls[0].to.startsWith('new')).toBe(true)
  })

  it('offsets the copies so they do not land on top of the originals', async () => {
    await duplicateGraph('core1', 48)
    expect(created.every((c) => c.x === 48 && c.y === 48)).toBe(true)
  })

  it('does not carry the original take history onto the copy', async () => {
    await duplicateGraph('core1')
    const copies = useMoodboardStore.getState().items.filter((i) => i.id.startsWith('new'))
    const copy = copies.find((i) => (i.data as { core?: unknown }).core)
    const core = (copy?.data as { core?: Record<string, unknown> } | undefined)?.core
    expect(core?.type).toBe('alibaba/z-image-turbo')
    expect(core?.params).toEqual({ steps: 8 })
    // The copy is a fresh slot, so it must not claim the original's renders.
    expect(core?.output).toBeUndefined()
  })
})

describe('the exported node', () => {
  it('types each param, so the folder is never guessed from its name', async () => {
    const core = (await graphRecipe('core1')).graph?.items.find((i) => i.id === 'core1')
    const params = (core?.data.core as { params: Record<string, { type: string; value: unknown }> })
      .params
    expect(params.steps).toEqual({ type: 'number', value: 8 })
  })

  it('carries the models beside the node that needs them, not in a list of its own', async () => {
    // A graph-wide list was built from params and from node defaults at once, so a node set to
    // klein-9b exported klein-4b beside it. Everything now derives from the node itself.
    const core = (await graphRecipe('core1')).graph?.items.find((i) => i.id === 'core1')
    const models = (core?.data.core as { models?: { name: string; directory: string }[] }).models
    expect(models?.map((m) => m.name)).toContain('qwen_3_4b.safetensors')
    expect(models?.find((m) => m.name === 'qwen_3_4b.safetensors')?.directory).toBe('text_encoders')
  })

  it('names the version, so a reader can tell the shapes apart', async () => {
    expect((await graphRecipe('core1')).version).toBe(2)
  })
})

describe('a node set to a non-default build', () => {
  it('does not export the default one beside it', async () => {
    // The bug this format replaced: a graph-wide list took the params and the node's declared
    // requirements at once, so a node on klein-9b carried klein-4b too.
    useModelRegistryStore.setState({
      entries: [
        {
          id: 'te-8b',
          filename: 'qwen_3_8b.safetensors',
          category: 'text_encoders',
          repo: 'org/repo',
          path: 'qwen_3_8b.safetensors',
          url: '',
        },
      ] as unknown as ReturnType<typeof useModelRegistryStore.getState>['entries'],
    })
    useCoreNodesStore.setState({
      descriptors: [
        {
          type: 'alibaba/z-image-turbo',
          title: 'Z',
          category: 'Generate',
          icon: 'wand',
          outputKind: 'image',
          inputs: [],
          outputs: [],
          params: [
            {
              key: 'text_encoder',
              label: 'Text encoder',
              widget: 'select',
              default: '',
              kind: 'model',
            },
          ],
        },
      ] as unknown as NodeDescriptor[],
    })
    useMoodboardStore.setState({
      items: [
        item('core9', 'core', {
          core: {
            type: 'alibaba/z-image-turbo',
            params: { text_encoder: 'qwen_3_8b.safetensors' },
          },
        }),
      ],
      connectors: [],
    })

    const core = (await graphRecipe('core9')).graph?.items[0]
    const models = (core?.data.core as { models?: { name: string }[] }).models ?? []
    expect(models.map((m) => m.name)).toEqual(['qwen_3_8b.safetensors'])
  })
})

describe('a param a wire is driving', () => {
  // A gen node's `model`/`vae`/`text_encoder` are params and input ports at once. Wired, the loader
  // wins at run time, so exporting the typed value listed the wrong checkpoint beside the right one.
  const GEN = item('gen1', 'core', {
    core: {
      type: 'krea/krea-2-turbo',
      params: { model: 'models/diffusion_models/krea2_turbo_bf16.safetensors' },
    },
  })
  const LOADER = item('load1', 'core', {
    core: { type: 'load/diffusion-model', params: { file: 'krea2_raw_bf16.safetensors' } },
  })

  function wire(from: string, to: string, handle: string): MoodboardConnector {
    return {
      ...connector(from, to),
      data: { sourceHandle: handle, targetHandle: handle },
    } as MoodboardConnector
  }

  const DESCRIPTORS = [
    {
      type: 'krea/krea-2-turbo',
      title: 'Krea 2 Turbo',
      category: 'Generate',
      icon: 'wand',
      outputKind: 'image',
      inputs: [{ id: 'model', label: 'Diffusion model', kind: 'model', required: false }],
      outputs: [],
      params: [{ key: 'model', label: 'Model', widget: 'select', default: '', kind: 'model' }],
    },
    {
      type: 'load/diffusion-model',
      title: 'Load Diffusion Model',
      category: 'Loaders',
      icon: 'box',
      outputKind: null,
      inputs: [],
      outputs: [{ id: 'model', label: 'Model', kind: 'model', required: false }],
      params: [{ key: 'file', label: 'File', widget: 'select', default: '', kind: 'model' }],
    },
  ] as unknown as NodeDescriptor[]

  beforeEach(() => {
    useMoodboardStore.setState({
      items: [GEN, LOADER],
      connectors: [wire('load1', 'gen1', 'model')],
    })
    useCoreNodesStore.setState({ descriptors: DESCRIPTORS })
  })

  it('is not exported as a model the graph needs', async () => {
    const recipe = await graphRecipe('gen1')
    const gen = recipe.graph!.items.find((i) => i.id === 'gen1')!
    const core = (gen.data as { core: { models?: { name: string }[] } }).core
    expect(core.models ?? []).toEqual([])
  })

  it('leaves the loader driving it to name the file', async () => {
    const recipe = await graphRecipe('gen1')
    const loader = recipe.graph!.items.find((i) => i.id === 'load1')!
    const core = (loader.data as { core: { models?: { name: string }[] } }).core
    expect((core.models ?? []).map((m) => m.name)).toEqual(['krea2_raw_bf16.safetensors'])
  })

  it('keeps the typed value in params, since that is what the node still holds', async () => {
    const recipe = await graphRecipe('gen1')
    const gen = recipe.graph!.items.find((i) => i.id === 'gen1')!
    const core = (gen.data as { core: { params: Record<string, { value: unknown }> } }).core
    expect(core.params.model.value).toBe('models/diffusion_models/krea2_turbo_bf16.safetensors')
  })

  it('exports a path-shaped pick under its bare filename', async () => {
    // A legacy full-path pick matched no registry row, so the same file appeared twice: once
    // correctly and once with an empty directory and no download link.
    useMoodboardStore.setState({ items: [GEN], connectors: [] })
    const recipe = await graphRecipe('gen1')
    const gen = recipe.graph!.items.find((i) => i.id === 'gen1')!
    const core = (gen.data as { core: { models?: { name: string }[] } }).core
    expect((core.models ?? []).map((m) => m.name)).toEqual(['krea2_turbo_bf16.safetensors'])
  })
})
