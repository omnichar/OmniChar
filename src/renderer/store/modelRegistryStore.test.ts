import { describe, expect, it } from 'vitest'
import type { MissingModel } from '@shared/types'
import { hasUndismissed, markPresent, preferredMatch } from './modelRegistryStore'

function model(id: string, filename: string) {
  return { id, label: filename, filename, category: 'vae', repo: 'r', path: filename }
}

const ROWS = [
  {
    wanted: 'a.safetensors',
    path: 'vae/a.safetensors',
    matches: [
      { model: model('a-fp8', 'a-fp8.safetensors'), exact: false, present: false },
      { model: model('a', 'a.safetensors'), exact: true, present: false },
    ],
  },
  {
    wanted: 'b.safetensors',
    path: 'vae/b.safetensors',
    matches: [{ model: model('b', 'b.safetensors'), exact: true, present: false }],
  },
] as unknown as MissingModel[]

describe('preferredMatch', () => {
  it('prefers the exact filename over another precision of the same file', () => {
    expect(preferredMatch(ROWS[0]!)?.model.id).toBe('a')
  })
})

describe('markPresent', () => {
  it('marks the row that just landed, so it stops offering Download', () => {
    // The popup's rows are the snapshot taken when it opened: without this a finished row fell back
    // to its Download button, the one action that is certainly wrong once the file is here.
    const next = markPresent(ROWS, 'a')!
    expect(preferredMatch(next[0]!)?.present).toBe(true)
    expect(preferredMatch(next[1]!)?.present).toBe(false)
  })

  it('leaves the untouched rows identical, so only the one that landed re-renders', () => {
    const next = markPresent(ROWS, 'a')!
    expect(next[1]).toBe(ROWS[1])
    expect(next[0]).not.toBe(ROWS[0])
  })

  it('is a no-op for a model this popup does not list', () => {
    expect(markPresent(ROWS, 'not-here')).toEqual(ROWS)
  })

  it('survives a closed popup', () => {
    expect(markPresent(null, 'a')).toBeNull()
  })
})

const row = (path: string): MissingModel => ({ wanted: path, path, matches: [] })

describe('missing-models popup', () => {
  it('stays shut once every missing file has been dismissed', () => {
    const dismissed = new Set(['vae/a.safetensors', 'text_encoders/b.safetensors'])
    expect(hasUndismissed([row('vae/a.safetensors')], dismissed)).toBe(false)
  })

  it('opens again for a file the user has not seen', () => {
    const dismissed = new Set(['vae/a.safetensors'])
    expect(hasUndismissed([row('vae/a.safetensors'), row('loras/c.safetensors')], dismissed)).toBe(
      true,
    )
  })

  it('opens the first time', () => {
    expect(hasUndismissed([row('vae/a.safetensors')], new Set())).toBe(true)
  })
})
