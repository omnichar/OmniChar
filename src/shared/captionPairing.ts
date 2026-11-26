// Pairs dataset captions to images; mirrored by Core's dataset_import.pair_captions, pinned by tests.

/** A filename's stem, lowercased, so `Cat.PNG` and `cat.txt` are one pair. */
export function captionStem(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? name
  return base.replace(/\.[^.]+$/, '').toLowerCase()
}

/** Each number in a stem, so `lin-01` offers 1. */
function numbersIn(stem: string): number[] {
  return (stem.match(/\d+/g) ?? []).map((n) => Number.parseInt(n, 10))
}

/** Image name -> caption: exact stem first, then a bare number (`1.txt`) when unique both ways. */
export function pairCaptions(
  imageNames: string[],
  captions: Map<string, string>,
): Map<string, string> {
  const paired = new Map<string, string>()
  const used = new Set<string>()
  for (const image of imageNames) {
    const stem = captionStem(image)
    const text = captions.get(stem)
    if (text !== undefined) {
      paired.set(image, text)
      used.add(stem)
    }
  }
  const imagesByNumber = new Map<number, string[]>()
  for (const image of imageNames) {
    if (paired.has(image)) continue
    for (const n of new Set(numbersIn(captionStem(image)))) {
      imagesByNumber.set(n, [...(imagesByNumber.get(n) ?? []), image])
    }
  }
  const claims = new Map<string, string[]>()
  for (const stem of captions.keys()) {
    if (used.has(stem) || !/^\d+$/.test(stem)) continue
    const candidates = imagesByNumber.get(Number.parseInt(stem, 10)) ?? []
    if (candidates.length === 1)
      claims.set(candidates[0], [...(claims.get(candidates[0]) ?? []), stem])
  }
  for (const [image, stems] of claims) {
    const text = stems.length === 1 ? captions.get(stems[0]) : undefined
    if (text !== undefined) paired.set(image, text)
  }
  return paired
}
