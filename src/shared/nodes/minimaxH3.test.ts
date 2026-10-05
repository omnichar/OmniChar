import { describe, expect, it } from 'vitest'
import { MINIMAX_H3_T2V } from './minimaxH3T2V'
import { MINIMAX_H3_I2V } from './minimaxH3I2V'
import { MINIMAX_H3_REF2V } from './minimaxH3Ref2V'
import { getNodeDef, modelOwnerLabel } from './registry'
import { defaultParams, emptyResolvedInputs, type ResolvedInputs } from './types'

const withImages = (images: string[]): ResolvedInputs => ({ ...emptyResolvedInputs(), images })
const byHandle = (map: Record<string, string[]>): ResolvedInputs => ({
  ...emptyResolvedInputs(),
  images: Object.values(map).flat(),
  byHandle: map,
})

describe('MiniMax H3 · text → video', () => {
  it('sends prompt, duration, 2K resolution and aspect ratio', () => {
    expect(MINIMAX_H3_T2V.resolveEndpoint(emptyResolvedInputs())).toBe('minimax/h3/text-to-video')
    const body = MINIMAX_H3_T2V.buildRequest(
      { ...defaultParams(MINIMAX_H3_T2V), prompt: 'a kitten' },
      emptyResolvedInputs(),
    )
    expect(body).toEqual({
      prompt: 'a kitten',
      duration: 5,
      resolution: '2K',
      aspect_ratio: '16:9',
    })
  })

  it('takes no inputs and declares no seed param (H3 has none)', () => {
    expect(MINIMAX_H3_T2V.inputs).toEqual([])
    expect(MINIMAX_H3_T2V.params.map((p) => p.key)).toEqual([
      'resolution',
      'duration',
      'aspect_ratio',
    ])
    expect(MINIMAX_H3_T2V.outputKind).toBe('video')
  })

  it('clamps duration to fal’s 5..15 range', () => {
    const at = (duration: number): unknown =>
      MINIMAX_H3_T2V.buildRequest({ prompt: 'p', duration }, emptyResolvedInputs()).duration
    expect(at(1)).toBe(5)
    expect(at(9)).toBe(9)
    expect(at(99)).toBe(15)
  })

  it('offers every resolution fal documents, not 2K alone', () => {
    // Pinning it to 2K charged a draft at the most expensive tier there is.
    const field = MINIMAX_H3_T2V.params.find((p) => p.key === 'resolution')
    expect(field?.widget === 'select' && field.options.map((o) => o.value)).toEqual([
      '480P',
      '768P',
      '2K',
      '4K',
    ])
    expect(field?.default).toBe('2K')
  })

  it('sends the picked resolution rather than a hardcoded one', () => {
    const body = MINIMAX_H3_T2V.buildRequest(
      { prompt: 'p', resolution: '480P' },
      emptyResolvedInputs(),
    )
    expect(body.resolution).toBe('480P')
    // A node saved before the param had options falls back rather than sending something invalid.
    expect(MINIMAX_H3_T2V.buildRequest({ prompt: 'p' }, emptyResolvedInputs()).resolution).toBe(
      '2K',
    )
    expect(
      MINIMAX_H3_T2V.buildRequest({ prompt: 'p', resolution: '8K' }, emptyResolvedInputs())
        .resolution,
    ).toBe('2K')
  })

  it('prices per second by resolution, from fal’s published rates', () => {
    const at = (resolution: string, duration = 5): number | undefined =>
      MINIMAX_H3_T2V.estimatePrice?.({ resolution, duration })?.amount
    expect(at('480P')).toBeCloseTo(0.25, 5)
    expect(at('768P')).toBeCloseTo(0.3, 5)
    expect(at('2K')).toBeCloseTo(0.65, 5)
    expect(at('4K')).toBeCloseTo(0.8, 5)
    expect(at('2K', 15)).toBeCloseTo(1.95, 5)
    expect(MINIMAX_H3_T2V.estimatePrice?.({ duration: 5 })?.approx).toBe(true)
  })
})

describe('MiniMax H3 · image → video', () => {
  it('declares a required start image and an optional end image', () => {
    expect(MINIMAX_H3_I2V.inputs).toEqual([
      { id: 'image', label: 'Image', kind: 'image', required: true },
      { id: 'end_image', label: 'End image', kind: 'image', required: false },
    ])
  })

  it('omits end_image_url when only a start frame is wired', () => {
    const body = MINIMAX_H3_I2V.buildRequest({ prompt: 'p' }, withImages(['data:start']))
    expect(body.image_url).toBe('data:start')
    expect(body.end_image_url).toBeUndefined()
  })

  it('routes each port by handle, not by insertion order', () => {
    const body = MINIMAX_H3_I2V.buildRequest(
      { prompt: 'p' },
      byHandle({ end_image: ['data:end'], image: ['data:start'] }),
    )
    expect(body.image_url).toBe('data:start')
    expect(body.end_image_url).toBe('data:end')
  })

  it('falls back to wiring order for untagged inputs', () => {
    const body = MINIMAX_H3_I2V.buildRequest(
      { prompt: 'p' },
      withImages(['data:start', 'data:end']),
    )
    expect(body.image_url).toBe('data:start')
    expect(body.end_image_url).toBe('data:end')
  })

  it('sends no aspect_ratio - H3 takes framing from the start image', () => {
    const body = MINIMAX_H3_I2V.buildRequest(
      { ...defaultParams(MINIMAX_H3_I2V), prompt: 'p' },
      withImages(['data:start']),
    )
    expect(body).toEqual({
      prompt: 'p',
      duration: 5,
      resolution: '2K',
      image_url: 'data:start',
    })
  })
})

describe('MiniMax H3 · reference → video', () => {
  it('maps each reference bucket and omits the empty ones', () => {
    const body = MINIMAX_H3_REF2V.buildRequest(
      { prompt: 'Image 1 is the lead' },
      { ...emptyResolvedInputs(), images: ['data:a', 'data:b'] },
    )
    expect(body.reference_image_urls).toEqual(['data:a', 'data:b'])
    expect(body.reference_video_urls).toBeUndefined()
    expect(body.reference_audio_urls).toBeUndefined()
    expect(body.aspect_ratio).toBe('adaptive')
  })

  it('carries video and audio references through', () => {
    const body = MINIMAX_H3_REF2V.buildRequest(
      { prompt: 'p' },
      {
        ...emptyResolvedInputs(),
        images: ['data:i'],
        videos: ['data:v'],
        audios: ['data:s'],
      },
    )
    expect(body.reference_image_urls).toEqual(['data:i'])
    expect(body.reference_video_urls).toEqual(['data:v'])
    expect(body.reference_audio_urls).toEqual(['data:s'])
  })

  it('trims to fal’s per-kind caps and the 12-file total', () => {
    const many = (prefix: string, n: number): string[] =>
      Array.from({ length: n }, (_, i) => `${prefix}${i}`)
    const body = MINIMAX_H3_REF2V.buildRequest(
      { prompt: 'p' },
      {
        ...emptyResolvedInputs(),
        images: many('i', 12),
        videos: many('v', 5),
        audios: many('s', 5),
      },
    )
    const images = body.reference_image_urls as string[]
    const videos = body.reference_video_urls as string[]
    expect(images).toHaveLength(9)
    expect(videos).toHaveLength(3)
    // 9 + 3 already fills the 12-file budget, so audio yields.
    expect(body.reference_audio_urls).toBeUndefined()
  })

  const voiced = (audios: string[]): ResolvedInputs => ({
    ...emptyResolvedInputs(),
    images: ['data:i'],
    audios,
    character: {
      name: 'Ada',
      refs: ['data:ref'],
      roles: ['face'],
      promptPrefix: '<Picture 2> shows Ada. <Audio 2> is Ada’s voice. ',
      voice: 'data:voice',
    },
  })

  it('appends a character’s voice after the wired audio', () => {
    expect(MINIMAX_H3_REF2V.character?.voicePort).toBe('reference_audio_urls')
    const body = MINIMAX_H3_REF2V.buildRequest({ prompt: 'p' }, voiced(['data:s']))
    expect(body.reference_audio_urls).toEqual(['data:s', 'data:voice'])
    expect(body.reference_image_urls).toEqual(['data:i', 'data:ref'])
  })

  it('cuts the voice first when the audio cap bites', () => {
    const body = MINIMAX_H3_REF2V.buildRequest({ prompt: 'p' }, voiced(['a', 'b', 'c']))
    expect(body.reference_audio_urls).toEqual(['a', 'b', 'c'])
  })

  it('sends no voice for a character without one', () => {
    const resolved = voiced(['data:s'])
    delete resolved.character?.voice
    const body = MINIMAX_H3_REF2V.buildRequest({ prompt: 'p' }, resolved)
    expect(body.reference_audio_urls).toEqual(['data:s'])
  })

  it('is the only H3 endpoint that takes a voice', () => {
    expect(MINIMAX_H3_T2V.character?.voicePort).toBeUndefined()
    expect(MINIMAX_H3_I2V.character?.voicePort).toBeUndefined()
  })
})

describe('H3 registry wiring', () => {
  it('registers all three endpoints under a MiniMax group', () => {
    for (const id of [
      'minimax/h3/text-to-video',
      'minimax/h3/image-to-video',
      'minimax/h3/reference-to-video',
    ]) {
      expect(getNodeDef(id)?.id).toBe(id)
      expect(modelOwnerLabel(id)).toBe('MiniMax')
    }
  })
})

describe('MiniMax H3 reference-image surcharge', () => {
  it('charges nothing extra for the first five references', () => {
    const base = MINIMAX_H3_REF2V.estimatePrice?.({ resolution: '2K', duration: 5 })?.amount
    for (const n of [0, 1, 5]) {
      expect(
        MINIMAX_H3_REF2V.estimatePrice?.({ resolution: '2K', duration: 5 }, { referenceImages: n })
          ?.amount,
      ).toBeCloseTo(base ?? 0, 5)
    }
  })

  it('adds $0.08 for each reference past the fifth', () => {
    // A character contributing its whole set is exactly the case params alone cannot see: nine
    // references is four chargeable ones, $0.32 the estimate used to miss.
    const at = (referenceImages: number): number | undefined =>
      MINIMAX_H3_REF2V.estimatePrice?.({ resolution: '2K', duration: 5 }, { referenceImages })
        ?.amount
    expect(at(6)).toBeCloseTo(0.65 + 0.08, 5)
    expect(at(9)).toBeCloseTo(0.65 + 0.32, 5)
  })

  it('leaves the endpoints without a reference port alone', () => {
    // Only reference-to-video bills per image; text and image to video have nothing to count.
    const wired = { referenceImages: 9 }
    expect(MINIMAX_H3_T2V.estimatePrice?.({ duration: 5 }, wired)?.amount).toBeCloseTo(0.65, 5)
    expect(MINIMAX_H3_I2V.estimatePrice?.({ duration: 5 }, wired)?.amount).toBeCloseTo(0.65, 5)
  })
})
