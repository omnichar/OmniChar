import { captionStem } from '@shared/captionPairing'

export const isCaptionFile = (f: File): boolean => f.name.toLowerCase().endsWith('.txt')

/** Every `.txt` among the files, as `stem -> caption`, ready for `pairCaptions`. */
export async function readCaptionFiles(files: File[]): Promise<Map<string, string>> {
  const captions = new Map<string, string>()
  await Promise.all(
    files.filter(isCaptionFile).map(async (f) => {
      const text = (await f.text()).trim()
      if (text) captions.set(captionStem(f.name), text)
    }),
  )
  return captions
}
