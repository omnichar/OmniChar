/**
 * fal.ai `minimax/h3/reference-to-video` - MiniMax H3 (Hailuo 03) reference-to-video, 2K.
 *
 * The prompt addresses references by modality and position ("Image 1 is the protagonist, Video 1 is
 * the camera move"), so wiring order is meaning. All references are optional, but fal rejects a set
 * that is audio only. See https://fal.ai/models/minimax/h3/reference-to-video.
 */
import { constantEndpoint, selectParam } from './builders'
import { H3_ASPECT_RATIOS, H3_BASE_PARAMS, buildH3Body, estimateH3Price } from './minimaxH3Shared'
import {
  portMedia,
  withCharacterPrompt,
  withCharacterRefs,
  withCharacterVoice,
  type NodeDef,
  type ResolvedInputs,
} from './types'

// fal's caps: 9 images, 3 videos, 3 audios, and no more than 12 files across all three. We trim to
// them rather than let an over-wired node fail at the API.
const MAX_IMAGES = 9
const MAX_VIDEOS = 3
const MAX_AUDIOS = 3
const MAX_TOTAL = 12

export const MINIMAX_H3_REF2V: NodeDef = {
  id: 'minimax/h3/reference-to-video',
  title: 'MiniMax H3 · Reference → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [
    { id: 'reference_image_urls', label: 'Images', kind: 'image[]', required: false },
    { id: 'reference_video_urls', label: 'Video', kind: 'video[]', required: false },
    { id: 'reference_audio_urls', label: 'Audio', kind: 'audio[]', required: false },
    { id: 'character', label: 'Character', kind: 'character', required: false },
  ],
  // The video model that will actually carry a character: measured against emmy-s500-v6, all five
  // references accepted, a face in every sampled frame and identity holding at 61 mean / 54 min.
  // `token` because H3 resolves `<Picture N>` and reads neither of the other two forms.
  character: {
    port: 'reference_image_urls',
    style: 'token',
    maxImages: MAX_IMAGES,
    maxRefs: MAX_IMAGES,
    voicePort: 'reference_audio_urls',
  },
  params: [
    ...H3_BASE_PARAMS,
    selectParam('aspect_ratio', 'Aspect ratio', ['adaptive', ...H3_ASPECT_RATIOS], 'adaptive'),
  ],
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('minimax/h3/reference-to-video'),
  buildRequest: (params, resolved: ResolvedInputs) => {
    const images = withCharacterRefs(MINIMAX_H3_REF2V, resolved, 'reference_image_urls').slice(
      0,
      MAX_IMAGES,
    )
    const videos = portMedia(MINIMAX_H3_REF2V, resolved, 'reference_video_urls').slice(
      0,
      MAX_VIDEOS,
    )
    // Images and video carry the subject, so audio yields first when the combined cap bites.
    const audios = withCharacterVoice(MINIMAX_H3_REF2V, resolved, 'reference_audio_urls')
      .slice(0, MAX_AUDIOS)
      .slice(0, Math.max(0, MAX_TOTAL - images.length - videos.length))
    return buildH3Body(
      { ...params, prompt: withCharacterPrompt(resolved, String(params.prompt ?? '')) },
      {
        aspect_ratio: String(params.aspect_ratio ?? 'adaptive'),
        ...(images.length > 0 ? { reference_image_urls: images } : {}),
        ...(videos.length > 0 ? { reference_video_urls: videos } : {}),
        ...(audios.length > 0 ? { reference_audio_urls: audios } : {}),
      },
    )
  },
  // The only H3 endpoint with a reference port, so the only one that bills per reference image.
  estimatePrice: (params, wired) => estimateH3Price(params, wired),
}
