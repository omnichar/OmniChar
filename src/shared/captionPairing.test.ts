import { describe, expect, it } from 'vitest'
import { pairCaptions } from './captionPairing'

const caps = (entries: [string, string][]): Map<string, string> => new Map(entries)

describe('pairCaptions', () => {
  it('pairs an exact stem, case-insensitively', () => {
    const out = pairCaptions(['Cat.PNG'], caps([['cat', 'a cat']]))
    expect(out.get('Cat.PNG')).toBe('a cat')
  })

  it('pairs numbered captions to descriptively named images', () => {
    const images = [
      'lin-01-front-studio.png',
      'lin-02-profile-studio.png',
      'lin-10-bw-sidelight.png',
    ]
    const out = pairCaptions(
      images,
      caps([
        ['1', 'front'],
        ['2', 'profile'],
        ['10', 'bw'],
      ]),
    )
    expect(out.get('lin-01-front-studio.png')).toBe('front')
    expect(out.get('lin-02-profile-studio.png')).toBe('profile')
    expect(out.get('lin-10-bw-sidelight.png')).toBe('bw')
  })

  it('prefers an exact stem over a number', () => {
    const out = pairCaptions(['1.png', 'shot-1.png'], caps([['1', 'one']]))
    expect(out.get('1.png')).toBe('one')
    expect(out.has('shot-1.png')).toBe(false)
  })

  it('leaves an ambiguous number unpaired rather than guessing', () => {
    const out = pairCaptions(['a-1.png', 'b-1.png'], caps([['1', 'which?']]))
    expect(out.size).toBe(0)
  })

  it('leaves an image two numbered captions could claim unpaired', () => {
    const out = pairCaptions(
      ['img-2024-05.png'],
      caps([
        ['2024', 'x'],
        ['5', 'y'],
      ]),
    )
    expect(out.size).toBe(0)
  })

  it('never pairs a caption with letters in it by number', () => {
    const out = pairCaptions(['lin-01.png'], caps([['note1', 'x']]))
    expect(out.size).toBe(0)
  })
})
