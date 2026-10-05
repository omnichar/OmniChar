/**
 * The single typed contract for the renderer ↔ main IPC bridge.
 *
 * - `IpcChannels` are the only channel strings allowed (no stringly-typed
 *   `invoke('something')` scattered around - see CLAUDE.md).
 * - `InlineStudioApi` is the exact backend surface the web client implements against Core; the
 *   preload. The renderer imports this type; the main process implements it.
 */
import type { AppliedCharacter, CharacterPromptStyle, CharacterRole } from './nodes/types'
import type {
  ActiveGeneration,
  Project,
  RecentProject,
  Asset,
  AssetFolder,
  CharacterSummary,
  CharacterProgressEvent,
  MissingModel,
  RegistryModel,
  MoodboardItem,
  MoodboardConnector,
  MoodboardSnapshot,
  MoodboardItemData,
  DirectorTimeline,
  TrimResolved,
  TimelineProgressEvent,
  GenerationProgressEvent,
  GenerationNodeDoneEvent,
  GenerationDoneEvent,
  GenerationErrorEvent,
  GenerationCancelledEvent,
  ActivityRun,
  ActivityChangedEvent,
  ModelTreeRoot,
  Frame,
  Take,
  FrameInput,
  AppSettings,
  ApiKeyStatus,
  UpdateAvailableEvent,
  UpdateProgressEvent,
  UpdateDownloadedEvent,
  ModelDownloadProgressEvent,
  ModelDownloadDoneEvent,
  ModelDownloadErrorEvent,
  TrainingDataset,
  DatasetRepoPreview,
  StagedDatasetItem,
  TrainingDatasetItem,
  TrainingMode,
  CaptionerModel,
  TrainingHyperparams,
  TrainingRun,
  TrainingProgressEvent,
  CaptionProgressEvent,
  TrainingLogEvent,
  SweepResult,
  TuneProgressEvent,
  TuneDoneEvent,
  TrainingSampleEvent,
  TrainingSnapshot,
  TrainingDoneEvent,
  TrainingNodeBoundEvent,
  TrainingErrorEvent,
  SystemStatsEvent,
  CoreStatus,
  ExportResult,
  ProjectExportResult,
  ProjectMediaDirs,
  WorkflowCatalogue,
  WorkflowDetail,
  WorkflowSort,
} from './types'
import type { Result } from './result'
import type { CoreModels, ModelRequirements } from './coreNodes'
import type {
  ExtensionInfo,
  ExtensionsStatus,
  InstallFailure,
  InstallOutcome,
  InstallProgressEvent,
  InstallSuccess,
  LifecycleResult,
  RegistryIndex,
  UpdateStatus,
} from './extensions'

export const IpcChannels = {
  project: {
    create: 'project:create',
    open: 'project:open',
    openDialog: 'project:openDialog',
    openZip: 'project:openZip',
    listRecent: 'project:listRecent',
    current: 'project:current',
    close: 'project:close',
    mediaDirs: 'project:mediaDirs',
    export: 'project:export',
  },
  clipboard: {
    writeText: 'clipboard:writeText',
  },
  assets: {
    importDialog: 'assets:importDialog',
    importPaths: 'assets:importPaths',
    list: 'assets:list',
    delete: 'assets:delete',
  },
  folders: {
    list: 'folders:list',
    create: 'folders:create',
    rename: 'folders:rename',
    delete: 'folders:delete',
  },
  frames: {
    list: 'frames:list',
    importAsFrames: 'frames:importAsFrames',
    addFromAsset: 'frames:addFromAsset',
    rename: 'frames:rename',
    reorder: 'frames:reorder',
    delete: 'frames:delete',
    clone: 'frames:clone',
    setHero: 'frames:setHero',
    listTakes: 'frames:listTakes',
    heroTakes: 'frames:heroTakes',
    listInputs: 'frames:listInputs',
    addInput: 'frames:addInput',
    addInputs: 'frames:addInputs',
    addSourceInput: 'frames:addSourceInput',
    removeInput: 'frames:removeInput',
    removeInputById: 'frames:removeInputById',
    reorderInputs: 'frames:reorderInputs',
    listAllTakes: 'frames:listAllTakes',
    deleteTake: 'frames:deleteTake',
    setFalParams: 'frames:setFalParams',
    setModel: 'frames:setModel',
    setProvider: 'frames:setProvider',
    resolveFalInputs: 'frames:resolveFalInputs',
  },
  generation: {
    /** Run a fal frame and its upstream chain (topologically). Streams progress via events. */
    run: 'generation:run',
    runWorkflow: 'generation:runWorkflow',
    /** Abort the in-flight generation run (optionally just one frame's). */
    cancel: 'generation:cancel',
    /** What is still running, so a reloaded page can rebuild its queue. */
    active: 'generation:active',
    /** Re-poll + finish any runs that were in flight when the app last closed. */
    resumePending: 'generation:resumePending',
  },
  activity: {
    /** Queued + running, across every project and both tabs. */
    list: 'activity:list',
    /** Finished runs for the open project. */
    history: 'activity:history',
    /** Cancel any run by its id, whether it is a Core, fal, or training run. */
    cancel: 'activity:cancel',
    clearHistory: 'activity:clearHistory',
  },
  training: {
    listDatasets: 'training:listDatasets',
    createDataset: 'training:createDataset',
    updateDataset: 'training:updateDataset',
    listItems: 'training:listItems',
    addItems: 'training:addItems',
    addFromPath: 'training:addFromPath',
    inspectDatasetPath: 'training:inspectDatasetPath',
    inspectDatasetRepo: 'training:inspectDatasetRepo',
    stageAssets: 'training:stageAssets',
    captionAssets: 'training:captionAssets',
    stageFromPath: 'training:stageFromPath',
    stageFromRepo: 'training:stageFromRepo',
    commitStaged: 'training:commitStaged',
    removeItem: 'training:removeItem',
    setCaption: 'training:setCaption',
    /** Pair a reference clip with an item (Control LoRA), or clear it with null. */
    setItemReference: 'training:setItemReference',
    /** Switch a dataset between clip and control training. */
    setDatasetMode: 'training:setDatasetMode',
    autoCaption: 'training:autoCaption',
    captioners: 'training:captioners',
    listRuns: 'training:listRuns',
    start: 'training:start',
    resume: 'training:resume',
    cancel: 'training:cancel',
    discard: 'training:discard',
    status: 'training:status',
    /** Mid-run LoRAs this run has written. */
    snapshots: 'training:snapshots',
    /** Copy one snapshot into models/loras/ so a Load LoRA node can pick it. */
    exportSnapshot: 'training:exportSnapshot',
  },
  falSettings: {
    status: 'falSettings:status',
    setApiKey: 'falSettings:setApiKey',
    clearApiKey: 'falSettings:clearApiKey',
  },
  hfSettings: {
    status: 'hfSettings:status',
    setToken: 'hfSettings:setToken',
    clearToken: 'hfSettings:clearToken',
  },
  settings: {
    get: 'settings:get',
    setCoreUrl: 'settings:setCoreUrl',
  },
  core: {
    status: 'core:status',
    models: 'core:models',
  },
  models: {
    /** The model components a node needs + whether each is on disk. */
    requirements: 'models:requirements',
    /** Explicitly download one component (by id) or `'all'` missing ones into models/. */
    download: 'models:download',
    /** Read-only listing of every models root on disk, for the Models panel. */
    tree: 'models:tree',
    /** Re-read the models roots so files added or removed on disk reach the pickers. */
    rescan: 'models:rescan',
    /** The published model list, for Settings and for offering a missing file. */
    registry: 'models:registry',
    /** Which of these filenames are absent, and what the registry offers for each. */
    resolveMissing: 'models:resolveMissing',
    /** Download one registry model by its id. */
    downloadRegistry: 'models:downloadRegistry',
    cancelRegistryDownload: 'models:cancelRegistryDownload',
    downloadQueue: 'models:downloadQueue',
  },
  workflows: {
    /** The published catalogue, proxied by Core with an ETag disk cache. */
    list: 'workflows:list',
    /** One workflow with its graph. Uncached upstream, so each call counts as a view. */
    detail: 'workflows:detail',
    /** Beacon an event the site cannot observe on its own, chiefly a completed import. */
    event: 'workflows:event',
    /** Remember that the popup has auto-opened for this project. */
    markPrompted: 'workflows:markPrompted',
  },
  extensions: {
    /** Installed extensions + whether the machine has the tools to install more. */
    status: 'ext:manage:status',
    list: 'ext:manage:list',
    /** Install from a git URL at a tag/branch/sha. Returns a consent pause or an outcome. */
    install: 'ext:manage:install',
    uninstall: 'ext:manage:uninstall',
    setEnabled: 'ext:manage:setEnabled',
    setNodeEnabled: 'ext:manage:setNodeEnabled',
    versions: 'ext:manage:versions',
    /** Roll back (or forward) to an already-installed version. Always needs a restart. */
    switchVersion: 'ext:manage:switchVersion',
    /** The published registry index, cached on disk so it works offline. */
    checkUpdates: 'ext:manage:checkUpdates',
    registryIndex: 'ext:manage:registryIndex',
  },
  export: {
    exportFrames: 'export:exportFrames',
  },
  characters: {
    /** Every saved character in `models/characters/`, newest first. */
    list: 'characters:list',
    delete: 'characters:delete',
    /** Turn a generated take into a character, so a good render becomes reusable. */
    createFromTake: 'characters:createFromTake',
    /** A character compiled for one fal endpoint: normalised refs + the prompt text naming them. */
    applyFal: 'characters:applyFal',
    /** A finished sweep's result, so a node still reports it after a reload. */
    sweepResult: 'characters:sweepResult',
  },
  moodboard: {
    list: 'moodboard:list',
    addAsset: 'moodboard:addAsset',
    addText: 'moodboard:addText',
    addFrameFromAsset: 'moodboard:addFrameFromAsset',
    addEmptyFrame: 'moodboard:addEmptyFrame',
    addFrameItem: 'moodboard:addFrameItem',
    addPreview: 'moodboard:addPreview',
    addLayer: 'moodboard:addLayer',
    addDirector: 'moodboard:addDirector',
    addTrim: 'moodboard:addTrim',
    addLoader: 'moodboard:addLoader',
    addControlSpace: 'moodboard:addControlSpace',
    addGenNode: 'moodboard:addGenNode',
    addPrompt: 'moodboard:addPrompt',
    addCoreNode: 'moodboard:addCoreNode',
    addTrainDataset: 'moodboard:addTrainDataset',
    addCaption: 'moodboard:addCaption',
    addTrainer: 'moodboard:addTrainer',
    addLossGraph: 'moodboard:addLossGraph',
    addLogTail: 'moodboard:addLogTail',
    addResource: 'moodboard:addResource',
    updateItem: 'moodboard:updateItem',
    deleteItem: 'moodboard:deleteItem',
    removeCoreOutput: 'moodboard:removeCoreOutput',
    importAndPlace: 'moodboard:importAndPlace',
    createConnector: 'moodboard:createConnector',
    deleteConnector: 'moodboard:deleteConnector',
    setConnectorVolume: 'moodboard:setConnectorVolume',
    replaceBoard: 'moodboard:replaceBoard',
  },
  timeline: {
    resolve: 'timeline:resolve',
    resolveTrim: 'timeline:resolveTrim',
    setVolumes: 'timeline:setVolumes',
    buildPreview: 'timeline:buildPreview',
    export: 'timeline:export',
  },
  dialog: {
    pickDirectory: 'dialog:pickDirectory',
  },
  media: {
    save: 'media:save',
    copyImage: 'media:copyImage',
  },
  shell: {
    openExternal: 'shell:openExternal',
  },
  updates: {
    /** Manually trigger a check for a newer published release. */
    check: 'updates:check',
    /** Quit and install a downloaded update (Windows/Linux). */
    quitAndInstall: 'updates:quitAndInstall',
  },
  app: {
    /** The running app version (from package.json). */
    version: 'app:version',
  },
  events: {
    /** Main → renderer: the asset library changed (e.g. a video poster/transcode is ready). */
    libraryChanged: 'events:libraryChanged',
    /** Main → renderer: director timeline render progress. */
    timelineProgress: 'events:timelineProgress',
    /** Main → renderer: fal generation lifecycle (per-node progress, node done, done, error). */
    generationProgress: 'events:generationProgress',
    generationNodeDone: 'events:generationNodeDone',
    generationDone: 'events:generationDone',
    generationError: 'events:generationError',
    /**
     * Main → renderer: a run was cancelled. Without this a node cancelled from anywhere but the
     * tab that started it keeps spinning forever.
     */
    generationCancelled: 'events:generationCancelled',
    /** Main → renderer: the live run list changed (queued, started, progressed, finished). */
    activityChanged: 'events:activityChanged',
    /** Main → renderer: the installed model set changed, so every open client should refetch. */
    modelsChanged: 'events:modelsChanged',
    /** Main → renderer: the character library changed (created, edited, deleted, imported). */
    charactersChanged: 'events:charactersChanged',
    /** Main → renderer: an encode's phases, streamed while the create/edit call is still open. */
    characterProgress: 'events:characterProgress',
    /** Main → renderer: explicit model-download lifecycle (the node's model popup). */
    modelDownloadProgress: 'events:modelDownloadProgress',
    modelDownloadDone: 'events:modelDownloadDone',
    modelDownloadError: 'events:modelDownloadError',
    /** Main → renderer: LoRA training lifecycle (per-run progress, sample preview, done, error). */
    trainingProgress: 'events:trainingProgress',
    trainingSample: 'events:trainingSample',
    /** Main → renderer: a mid-run LoRA landed on disk. */
    trainingSnapshot: 'events:trainingSnapshot',
    trainingLog: 'events:trainingLog',
    captionProgress: 'events:captionProgress',
    trainingDone: 'events:trainingDone',
    trainingNodeBound: 'events:trainingNodeBound',
    tuneProgress: 'events:tuneProgress',
    tuneLog: 'events:tuneLog',
    tuneDone: 'events:tuneDone',
    tuneError: 'events:tuneError',
    tuneNodeBound: 'events:tuneNodeBound',
    trainingError: 'events:trainingError',
    /** Main → renderer: periodic host + GPU telemetry for the Trainer tab. */
    systemStats: 'events:systemStats',
    /** Main → renderer: extension install lifecycle (the Extensions dialog). */
    extensionInstallProgress: 'events:extensionInstallProgress',
    extensionInstallDone: 'events:extensionInstallDone',
    extensionInstallError: 'events:extensionInstallError',
    /** Main → renderer: auto-update lifecycle. */
    updateAvailable: 'events:updateAvailable',
    updateProgress: 'events:updateProgress',
    updateDownloaded: 'events:updateDownloaded',
  },
} as const

/** Geometry/content fields a moodboard item update may change. */
export interface MoodboardItemPatch {
  x?: number
  y?: number
  width?: number
  height?: number
  rotation?: number
  zIndex?: number
  data?: MoodboardItemData
  /** Containing layer id, or null to detach from any layer. */
  parentId?: string | null
}

export interface CreateFolderInput {
  name: string
  /** Parent folder id, or null for a root-level folder. */
  parentId: string | null
}

export interface CreateTrainingDatasetInput {
  name: string
  /** Optional trigger token injected into every caption. */
  triggerWord?: string
}

/** A rename, a new trigger word, or both. An omitted field is left as it is. */
export interface UpdateTrainingDatasetInput {
  name?: string
  triggerWord?: string
}

/** A fal frame's inputs resolved for building its request: media as data URIs + the prompt text. */
export interface ResolvedFalInputs {
  images: string[]
  videos: string[]
  audios: string[]
  /** The same URIs keyed by the input port each was wired to; untagged inputs are absent. */
  byHandle: Record<string, string[]>
  prompt: string | null
  /** The `.char` filename wired into the node's Character port, or null when none is. */
  character: string | null
}

/** What one endpoint needs a character normalised to, so `characters:applyFal` can compile it. */
export interface FalCharacterRequest {
  /** The `.char` filename, as `ResolvedFalInputs.character` reported it. */
  file: string
  /** Reference slots left for the character after the wired images. */
  limit: number
  /** The position its first reference lands on in the request's image list, 1-based. */
  firstPosition: number
  /** How this endpoint addresses a reference in the prompt. */
  style: CharacterPromptStyle
  /** Which roles this endpoint accepts. Absent means all of them. */
  keepRoles?: CharacterRole[]
  /** Name each role in the prompt, for an endpoint sent only part of the character. */
  roleLines?: boolean
  /** Offer the character's voice at this 1-based audio position. Core sends it only when `prompt`
   *  has dialogue. Absent sends none. */
  voice?: { firstPosition: number; prompt: string }
}

/** A prebuilt fal request the browser hands to the backend to run (fal defs are studio-side). */
export interface FalRunRequest {
  endpoint: string
  body: Record<string, unknown>
  outputKind: 'image' | 'video' | 'audio'
  /** The character applied, so the relay can score what fal returns against it. */
  characterFile?: string
}

export interface CreateProjectInput {
  /** Display name; also used to derive the `.inlinestudio` folder name. */
  name: string
  /** Absolute parent directory the `.inlinestudio` folder is created in. */
  parentDir: string
}

/** The backend API surface - implemented by the web client (createWebClient) against Inline Core. */
export interface InlineStudioApi {
  project: {
    create(input: CreateProjectInput): Promise<Result<Project>>
    /** Open a `.inlinestudio` folder by absolute path. */
    open(path: string): Promise<Result<Project>>
    /** Show a native folder picker and open the chosen project. */
    openDialog(): Promise<Result<Project | null>>
    /** Show a .zip picker, extract the exported project beside the zip, and open it. */
    openZip(): Promise<Result<Project | null>>
    listRecent(): Promise<Result<RecentProject[]>>
    current(): Promise<Result<Project | null>>
    /** Close the open project, so Core does not reopen it on its next start. */
    close(): Promise<Result<void>>
    /** Absolute input/output dirs of the open project, for sharing with ComfyUI. */
    mediaDirs(): Promise<Result<ProjectMediaDirs>>
    /** Zip a project folder (by path) into a portable .zip; null if the save dialog is cancelled. */
    export(path: string): Promise<Result<ProjectExportResult | null>>
  }
  clipboard: {
    writeText(text: string): Promise<Result<void>>
  }
  assets: {
    /**
     * Show a native multi-file picker, copy chosen media into the project's library
     * under `folderId` (null = root), and return the new rows.
     */
    importDialog(folderId: string | null): Promise<Result<Asset[]>>
    /** Import media by absolute path (e.g. OS files dropped onto the canvas). */
    importPaths(paths: string[], folderId: string | null): Promise<Result<Asset[]>>
    /** All assets in the open project, newest first. */
    list(): Promise<Result<Asset[]>>
    /** Delete an asset (file + row); blocked if used by a frame. */
    delete(assetId: string): Promise<Result<void>>
  }
  folders: {
    /** All asset folders in the open project. */
    list(): Promise<Result<AssetFolder[]>>
    create(input: CreateFolderInput): Promise<Result<AssetFolder>>
    rename(id: string, name: string): Promise<Result<AssetFolder>>
    /** Delete a folder; its assets and subfolders move up to the parent. */
    delete(id: string): Promise<Result<void>>
  }
  frames: {
    /** All frames in the open project, in order. */
    list(): Promise<Result<Frame[]>>
    /** Import media via dialog and create a frame per file. */
    importAsFrames(): Promise<Result<Frame[]>>
    /** Create a frame from an existing library asset. */
    addFromAsset(assetId: string): Promise<Result<Frame>>
    rename(id: string, name: string): Promise<Result<Frame>>
    /** Persist a new left-to-right ordering. */
    reorder(orderedIds: string[]): Promise<Result<void>>
    delete(id: string): Promise<Result<void>>
    /** Duplicate a frame (its inputs). */
    clone(id: string): Promise<Result<Frame>>
    /** Choose which take is the frame's Output (null clears it). */
    setHero(id: string, takeId: string | null): Promise<Result<Frame>>
    /** The frame's generated takes, newest first. */
    listTakes(frameId: string): Promise<Result<Take[]>>
    /** The hero (Output) take of every frame that has one. */
    heroTakes(): Promise<Result<Take[]>>
    /** All frame inputs across the project (group by frameId in the renderer). */
    listInputs(): Promise<Result<FrameInput[]>>
    /** Append a library asset as an input of the frame, optionally tagged with an input port id. */
    addInput(frameId: string, assetId: string, handle?: string | null): Promise<Result<FrameInput>>
    /** Append several library assets as inputs at once (atomic; skips duplicates). */
    addInputs(
      frameId: string,
      assetIds: string[],
      handle?: string | null,
    ): Promise<Result<FrameInput[]>>
    /** Link another frame's output as an input (resolves to its hero take). */
    addSourceInput(
      frameId: string,
      sourceFrameId: string,
      handle?: string | null,
    ): Promise<Result<FrameInput>>
    /** Remove an input by its library asset id (Frame Inspector). */
    removeInput(frameId: string, assetId: string): Promise<Result<void>>
    /** Remove one input by its row id - works for asset AND flow-link inputs. */
    removeInputById(frameId: string, inputId: string): Promise<Result<void>>
    /** Persist a new input ordering for the frame. */
    reorderInputs(frameId: string, orderedAssetIds: string[]): Promise<Result<void>>
    /** All takes across the project (group by frameId in the renderer). */
    listAllTakes(): Promise<Result<Take[]>>
    /** Delete a generated take (clears it as hero if it was). */
    deleteTake(takeId: string): Promise<Result<void>>
    /** Persist a fal frame's param values (from the GenNode widgets). Returns the updated frame. */
    setFalParams(frameId: string, params: Record<string, unknown>): Promise<Result<Frame>>
    /** Switch a fal frame to a different model (resets params + output kind). Returns the frame. */
    setModel(frameId: string, modelId: string): Promise<Result<Frame>>
    /**
     * Resolve a frame to the `fal` engine (a declarative model - `modelId` defaults to the first
     * registered model). Returns the frame.
     */
    setProvider(frameId: string, provider: 'fal', modelId?: string): Promise<Result<Frame>>
    /** Resolve a fal frame's inputs (media as data URIs) + prompt, for building its request. */
    resolveFalInputs(frameId: string): Promise<Result<ResolvedFalInputs>>
  }
  generation: {
    /**
     * Run a fal frame. On the single-process (web) backend the browser passes a prebuilt `request`
     * (fal defs are studio-side); the Electron backend builds it server-side and ignores `request`.
     * Resolves immediately; progress arrives via events.
     */
    run(frameId: string, request?: FalRunRequest): Promise<Result<void>>
    runWorkflow(itemId: string): Promise<Result<void>>
    /** Abort the in-flight run - a specific frame's, or all when no id is given. */
    cancel(frameId?: string): Promise<Result<void>>
    /**
     * The runs Core still has in flight. A page refresh throws away the renderer's copy of the
     * queue while the backend keeps working, so this is what rebuilds it on mount.
     */
    active(): Promise<Result<ActiveGeneration[]>>
    /** Re-poll + finish any generations that were in flight when the app last closed. */
    resumePending(): Promise<Result<void>>
  }
  activity: {
    /**
     * Every run Core still has queued or running, across projects and both tabs, including ones
     * submitted straight to the Core API rather than started here.
     */
    list(): Promise<Result<ActivityRun[]>>
    /** Finished runs for the open project, newest first. Empty when no project is open. */
    history(limit?: number): Promise<Result<ActivityRun[]>>
    /** Cancel a run by id; routes to the Core, fal, or training machinery as needed. */
    cancel(runId: string): Promise<Result<void>>
    clearHistory(): Promise<Result<void>>
  }
  training: {
    /** All training datasets in the open project. */
    listDatasets(): Promise<Result<TrainingDataset[]>>
    createDataset(input: CreateTrainingDatasetInput): Promise<Result<TrainingDataset>>
    /**
     * Rename a dataset or change its trigger word. The trigger is prepended to every caption when
     * a run exports its dataset, not stored on the items, so a change here lands on the next run
     * and leaves finished ones alone.
     */
    updateDataset(
      datasetId: string,
      input: UpdateTrainingDatasetInput,
    ): Promise<Result<TrainingDataset>>
    /** A dataset's items (images + captions), in order. */
    listItems(datasetId: string): Promise<Result<TrainingDatasetItem[]>>
    /**
     * Append library assets as dataset items (skips duplicates) and return the dataset's items.
     * Assets that name each other are paired: `bear_reference.mp4` becomes the reference of
     * `bear.mp4` rather than an item of its own, and a fully paired set switches to control mode.
     */
    addItems(datasetId: string, assetIds: string[]): Promise<Result<TrainingDatasetItem[]>>
    /**
     * Import every image and clip in a folder on the machine running Core, with any
     * `NNNN.txt` sidecar as that item's caption. Returns the dataset's full item list.
     */
    addFromPath(datasetId: string, path: string): Promise<Result<TrainingDatasetItem[]>>
    /**
     * What pulling a Hugging Face dataset would get, without pulling it. A video set is tens of
     * gigabytes, so the count and size are shown for confirmation first.
     */
    /** What importing a folder would get, before committing to it. */
    inspectDatasetPath(path: string): Promise<Result<DatasetRepoPreview>>
    inspectDatasetRepo(repo: string): Promise<Result<DatasetRepoPreview>>
    /**
     * Import a folder's media as library assets and work out the pairing, without touching any
     * dataset. Staging is what lets the Trainer show what a source holds before it is accepted.
     */
    /** Pair already-imported assets by name, without touching any dataset. */
    stageAssets(assetIds: string[]): Promise<Result<StagedDatasetItem[]>>
    /** Caption assets directly, for rows not yet in a dataset. Asset id -> caption. */
    captionAssets(assetIds: string[], model?: string): Promise<Result<Record<string, string>>>
    stageFromPath(path: string): Promise<Result<StagedDatasetItem[]>>
    /** The same for a Hugging Face dataset, downloading it into the project first. */
    stageFromRepo(repo: string): Promise<Result<StagedDatasetItem[]>>
    /** Write staged rows into a dataset. The only call that changes what will be trained. */
    commitStaged(
      datasetId: string,
      rows: StagedDatasetItem[],
    ): Promise<Result<TrainingDatasetItem[]>>
    removeItem(itemId: string): Promise<Result<void>>
    setCaption(itemId: string, caption: string): Promise<Result<TrainingDatasetItem>>
    /**
     * Pair a reference clip with an item, or clear it by passing null. Separate from `addItems`
     * because the two halves arrive separately: targets are dropped in first and paired afterwards,
     * so an unpaired item has to be a legible state rather than a failed import.
     */
    setItemReference(
      itemId: string,
      referenceAssetId: string | null,
    ): Promise<Result<TrainingDatasetItem>>
    /** Switch a dataset between clip and control training. */
    setDatasetMode(datasetId: string, mode: TrainingMode): Promise<Result<TrainingDataset>>
    /**
     * Auto-caption items with the local captioner; `overwrite` re-captions ones that already have
     * one. `model` picks a captioner (a `CaptionerModel.id` or a raw HF repo); omit for the default.
     */
    autoCaption(
      datasetId: string,
      overwrite: boolean,
      model?: string,
    ): Promise<Result<TrainingDatasetItem[]>>
    /** The caption models the UI can offer, first is the default. */
    captioners(): Promise<Result<CaptionerModel[]>>
    /** All training runs in the open project, newest first. */
    listRuns(): Promise<Result<TrainingRun[]>>
    /** Start a run over a dataset. Resolves immediately; progress arrives via `onTraining*`. */
    start(datasetId: string, hyperparams: TrainingHyperparams): Promise<Result<TrainingRun>>
    /** Resume an `interrupted` run from its last checkpoint. */
    resume(runId: string): Promise<Result<TrainingRun>>
    /** Cancel an in-flight run (saves a final checkpoint before exit). */
    cancel(runId: string): Promise<Result<void>>
    /**
     * Delete a run's checkpoints and working dir, making it unresumable. Used when the node's
     * hyperparameters change: a checkpoint encodes the rank, targets and base it was built with,
     * so resuming it would train something other than what the panel now says.
     */
    discard(runId: string): Promise<Result<TrainingRun>>
    /** One run's current durable state. */
    status(runId: string): Promise<Result<TrainingRun>>
    /** Every mid-run LoRA this run has written, oldest first. */
    snapshots(runId: string): Promise<Result<TrainingSnapshot[]>>
    /**
     * Copy one snapshot into `models/loras/` so a Load LoRA node can select it. Snapshots live in
     * the project's working dir, which no model picker scans. Done automatically as each snapshot
     * is written, so this is a retry for a run that predates that, or whose copy failed.
     */
    exportSnapshot(runId: string, step: number): Promise<Result<{ path: string }>>
  }
  falSettings: {
    /** Is a fal API key saved, and is it stored encrypted? */
    status(): Promise<Result<ApiKeyStatus>>
    /** Store the fal API key (encrypted via safeStorage when available). */
    setApiKey(key: string): Promise<Result<ApiKeyStatus>>
    /** Forget the stored fal key. */
    clearApiKey(): Promise<Result<ApiKeyStatus>>
  }
  hfSettings: {
    /** Is a Hugging Face token saved? Needed for gated repos such as FLUX.2 Klein 9B. */
    status(): Promise<Result<ApiKeyStatus>>
    /** Store the token, published as `HF_TOKEN` so every download path picks it up. */
    setToken(token: string): Promise<Result<ApiKeyStatus>>
    /** Forget the stored token. */
    clearToken(): Promise<Result<ApiKeyStatus>>
  }
  settings: {
    get(): Promise<Result<AppSettings>>
    setCoreUrl(url: string): Promise<Result<AppSettings>>
  }
  core: {
    status(): Promise<Result<CoreStatus>>
    models(): Promise<Result<CoreModels>>
  }
  models: {
    /** The model components a node needs + whether each is present under models/.
     * `params` because Train LoRA's base checkpoint follows the architecture its settings pick. */
    requirements(
      nodeType: string,
      params?: Record<string, unknown>,
    ): Promise<Result<ModelRequirements>>
    /** Download one component (its `id`) or `'all'` missing ones into models/. Fire-and-forget;
     * progress arrives on `events:modelDownload*`. */
    download(
      nodeType: string,
      componentId: string,
      params?: Record<string, unknown>,
    ): Promise<Result<void>>
    /** Every models root on disk as a read-only tree. No file actions. */
    tree(): Promise<Result<ModelTreeRoot[]>>
    /**
     * Re-scan the models roots and return the new registry version. The catalog caches its scan,
     * so a weight file dropped in by hand is invisible to the pickers until this runs.
     */
    rescan(): Promise<Result<{ registryVersion: string }>>
    /** Every published model, and whether the list came from a reachable registry. */
    registry(refresh?: boolean): Promise<Result<{ entries: RegistryModel[]; stale: boolean }>>
    /** Absent files and what the registry offers. Items may name the folder they belong in. */
    resolveMissing(
      wanted: (string | { filename: string; category?: string })[],
      refresh?: boolean,
    ): Promise<Result<{ missing: MissingModel[]; stale: boolean }>>
    /** Fetch a registry model. Progress arrives on the model-download events. */
    downloadRegistry(modelId: string): Promise<Result<void>>
    /** Drop a queued download, or stop the running one at its next chunk. */
    cancelRegistryDownload(modelId: string): Promise<Result<void>>
    /** What is downloading and what is waiting, so a reopened popup shows the same thing. */
    downloadQueue(): Promise<Result<{ active: string | null; queued: string[] }>>
  }
  workflows: {
    /**
     * The published catalogue. `refresh` forces a conditional GET; without it a cached copy is
     * served as-is. An unreachable site degrades to the saved entries marked `stale`.
     */
    list(sort?: WorkflowSort, refresh?: boolean): Promise<Result<WorkflowCatalogue>>
    /** One workflow with its graph. Null when the site is unreachable and nothing is cached. */
    detail(slug: string): Promise<Result<WorkflowDetail | null>>
    /** Fire-and-forget; a counter that did not move never surfaces as an error. */
    event(slug: string, event: 'import'): Promise<Result<void>>
    markPrompted(): Promise<Result<void>>
  }
  extensions: {
    /** Installed extensions + whether git/uv are available to install more. */
    status(): Promise<Result<ExtensionsStatus>>
    list(): Promise<Result<ExtensionInfo[]>>
    /**
     * Install from a git URL at `ref` - a tag, a branch, or `latest` to resolve the newest
     * release tag. Returns `needsConsent` (nothing installed) when the scan
     * raised HIGH/MEDIUM findings; re-call with the report's `consentRules` to proceed.
     * Progress arrives on `events:extensionInstall*`.
     */
    install(source: string, ref?: string, consents?: string[]): Promise<Result<InstallOutcome>>
    uninstall(extensionId: string): Promise<Result<LifecycleResult>>
    setEnabled(extensionId: string, enabled: boolean): Promise<Result<LifecycleResult>>
    setNodeEnabled(
      extensionId: string,
      nodeType: string,
      enabled: boolean,
    ): Promise<Result<LifecycleResult>>
    versions(
      extensionId: string,
    ): Promise<Result<{ extensionId: string; current: string; versions: string[] }>>
    /** Point a extension at an already-installed version. Needs a restart to take effect. */
    switchVersion(extensionId: string, version: string): Promise<Result<LifecycleResult>>
    /** Network-bound: asks each origin what its ref points at now. Best-effort. */
    checkUpdates(): Promise<Result<UpdateStatus[]>>
    registryIndex(refresh?: boolean): Promise<Result<RegistryIndex>>
  }
  export: {
    /** Pick a folder and write each frame's Output in order; null if cancelled. */
    exportFrames(): Promise<Result<ExportResult | null>>
  }
  characters: {
    /** Every saved character, newest first. An unreadable file is listed with `error` set. */
    list(): Promise<Result<CharacterSummary[]>>
    delete(file: string): Promise<Result<boolean>>
    /** Save a generated take as a new character. Creating, editing and building one otherwise
     * happen on the canvas, through the character nodes. */
    createFromTake(takeId: string, name: string): Promise<Result<CharacterSummary>>
    /** Compile a character for one fal endpoint. Composed in Core so reference ordering, the role
     * split and the position numbering stay where they already live. */
    applyFal(request: FalCharacterRequest): Promise<Result<AppliedCharacter>>
    sweepResult(runId: string): Promise<Result<SweepResult>>
  }
  moodboard: {
    /** The board's items + connectors. */
    list(): Promise<Result<MoodboardSnapshot>>
    /** Place an existing library asset on the board at (x, y). */
    addAsset(assetId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a new editable text item at (x, y). */
    addText(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a frame from a library asset AND place a frame node on the canvas. */
    addFrameFromAsset(assetId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create an empty frame AND place a frame node on the canvas at (x, y). */
    addEmptyFrame(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Place an existing frame as a node on the canvas. */
    addFrameItem(frameId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add an empty Preview node at (x, y). */
    addPreview(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a resizable layer group container at (x, y). */
    addLayer(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a video-director node (timeline-in-a-node) at (x, y). */
    addDirector(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add an "Edit Video/Audio" (trim) node at (x, y). */
    addTrim(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a standalone "Load Assets" node (holds library asset refs in its data). */
    addLoader(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a "Control Space" 3D pose-editor node (renders an OpenPose control map). */
    addControlSpace(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a fal generation frame for `modelId` AND place its node on the canvas at (x, y). */
    addGenNode(modelId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a text-prompt node (feeds a Generate node's prompt input) at (x, y). */
    addPrompt(x: number, y: number): Promise<Result<MoodboardItem>>
    addCoreNode(coreType: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Pick a training dataset and feed it downstream. */
    addTrainDataset(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Auto-caption a dataset's images. */
    addCaption(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Run a LoRA training job (run/stop/resume). */
    addTrainer(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Plot a run's loss curve. */
    addLossGraph(x: number, y: number): Promise<Result<MoodboardItem>>
    /** A log tail that shows whatever run is wired into it. */
    addLogTail(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Utility: read-only host telemetry node. */
    addResource(x: number, y: number): Promise<Result<MoodboardItem>>
    updateItem(id: string, patch: MoodboardItemPatch): Promise<Result<MoodboardItem>>
    deleteItem(id: string): Promise<Result<void>>
    /** Remove one render from a Core node's output history and unlink its file. */
    removeCoreOutput(itemId: string, takeId: string): Promise<Result<void>>
    /** Import media into the shared library AND place it on the board near (x, y). */
    importAndPlace(x: number, y: number): Promise<Result<MoodboardItem[]>>
    createConnector(
      fromItemId: string,
      toItemId: string,
      sourceHandle?: string | null,
      targetHandle?: string | null,
    ): Promise<Result<MoodboardConnector>>
    deleteConnector(id: string): Promise<Result<void>>
    /** Set a connector's per-input audio volume (0..1) - director L1 inputs. */
    setConnectorVolume(id: string, volume: number): Promise<Result<void>>
    /** Replace the board (used by canvas undo/redo). */
    replaceBoard(items: MoodboardItem[], connectors: MoodboardConnector[]): Promise<Result<void>>
  }
  timeline: {
    /** The derived timeline (video + L2 audio + volumes) for a director node. */
    resolve(ownerItemId: string): Promise<Result<DirectorTimeline>>
    /** The full source media behind a trim node's input (for its UI); null if unwired. */
    resolveTrim(itemId: string): Promise<Result<TrimResolved | null>>
    /** Set the L1/L2 layer volumes (0..1) on a director node. */
    setVolumes(ownerItemId: string, l1Volume: number, l2Volume: number): Promise<Result<void>>
    /** Render a low-res proxy preview; returns its project-relative path (null if empty). */
    buildPreview(ownerItemId: string): Promise<Result<string | null>>
    /** Export the timeline to a user-chosen MP4; returns the path written (null if cancelled). */
    export(ownerItemId: string): Promise<Result<string | null>>
    /** Subscribe to render progress (0..1). Returns an unsubscribe fn. */
    onProgress(cb: (e: TimelineProgressEvent) => void): () => void
  }
  dialog: {
    /** Native folder picker; returns the chosen absolute path or null if cancelled. */
    pickDirectory(): Promise<Result<string | null>>
  }
  media: {
    /**
     * Save a project media file (an image/video/audio take or asset) to a location
     * the user picks. `src` is its media URL or project-relative path; `suggestedName`
     * seeds the save dialog's filename (the source extension is appended if missing).
     * Resolves to `true` if saved, `false` if the dialog was cancelled.
     */
    save(src: string, suggestedName: string): Promise<Result<boolean>>
    /** Copy an image (by media URL or project-relative path) to the system clipboard. */
    copyImage(src: string): Promise<Result<void>>
  }
  shell: {
    /** Open an http(s) URL in the user's default browser. */
    openExternal(url: string): Promise<Result<void>>
  }
  updates: {
    /** Trigger a check for a newer published release. */
    check(): Promise<Result<void>>
    /** Quit and install a downloaded update (Windows/Linux). */
    quitAndInstall(): Promise<Result<void>>
  }
  app: {
    /** The running app version (from package.json). */
    version(): Promise<Result<string>>
  }
  /** Resolve the absolute path of a File dropped from the OS (Electron webUtils). Sync. */
  getPathForFile(file: File): string
  events: {
    /** Subscribe to "asset library changed" pushes from main. Returns an unsubscribe fn. */
    onLibraryChanged(callback: () => void): () => void
    /** Subscribe to fal generation lifecycle pushes. Each returns an unsubscribe fn. */
    onGenerationProgress(callback: (e: GenerationProgressEvent) => void): () => void
    onGenerationNodeDone(callback: (e: GenerationNodeDoneEvent) => void): () => void
    onGenerationDone(callback: (e: GenerationDoneEvent) => void): () => void
    onGenerationError(callback: (e: GenerationErrorEvent) => void): () => void
    onGenerationCancelled(callback: (e: GenerationCancelledEvent) => void): () => void
    onActivityChanged(callback: (e: ActivityChangedEvent) => void): () => void
    onModelsChanged(callback: (e: { registryVersion: string }) => void): () => void
    /** Subscribe to "the character library changed" pushes. Returns an unsubscribe fn. */
    onCharactersChanged(callback: () => void): () => void
    /** Subscribe to encode-progress pushes. Returns an unsubscribe fn. */
    onCharacterProgress(callback: (e: CharacterProgressEvent) => void): () => void
    /** Subscribe to explicit model-download lifecycle pushes. Each returns an unsubscribe fn. */
    onModelDownloadProgress(callback: (e: ModelDownloadProgressEvent) => void): () => void
    onModelDownloadDone(callback: (e: ModelDownloadDoneEvent) => void): () => void
    onModelDownloadError(callback: (e: ModelDownloadErrorEvent) => void): () => void
    /** Subscribe to LoRA training lifecycle pushes. Each returns an unsubscribe fn. */
    onTrainingProgress(callback: (e: TrainingProgressEvent) => void): () => void
    onTrainingSample(callback: (e: TrainingSampleEvent) => void): () => void
    onTrainingSnapshot(callback: (e: TrainingSnapshot) => void): () => void
    /** One stdout line from the trainer subprocess. */
    onTrainingLog(callback: (e: TrainingLogEvent) => void): () => void
    /** Auto-caption progress for a dataset. */
    onCaptionProgress(callback: (e: CaptionProgressEvent) => void): () => void
    onTrainingDone(callback: (e: TrainingDoneEvent) => void): () => void
    /** A Train LoRA node bound to the run its graph just started. */
    onTrainingNodeBound(callback: (e: TrainingNodeBoundEvent) => void): () => void
    onTuneProgress(callback: (e: TuneProgressEvent) => void): () => void
    onTuneLog(callback: (e: TrainingLogEvent) => void): () => void
    onTuneDone(callback: (e: TuneDoneEvent) => void): () => void
    onTuneError(callback: (e: { runId: string; error: string }) => void): () => void
    onTuneNodeBound(callback: (e: TrainingNodeBoundEvent) => void): () => void
    onTrainingError(callback: (e: TrainingErrorEvent) => void): () => void
    /** Subscribe to periodic host + GPU telemetry. Returns an unsubscribe fn. */
    onSystemStats(callback: (e: SystemStatsEvent) => void): () => void
    onExtensionInstallProgress(callback: (e: InstallProgressEvent) => void): () => void
    onExtensionInstallDone(callback: (e: InstallSuccess) => void): () => void
    onExtensionInstallError(callback: (e: InstallFailure) => void): () => void
    /** Subscribe to auto-update lifecycle pushes. Each returns an unsubscribe fn. */
    onUpdateAvailable(callback: (e: UpdateAvailableEvent) => void): () => void
    onUpdateProgress(callback: (e: UpdateProgressEvent) => void): () => void
    onUpdateDownloaded(callback: (e: UpdateDownloadedEvent) => void): () => void
  }
}
