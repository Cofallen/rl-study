import fs from 'node:fs'
import { execFileSync } from 'node:child_process'

const pngSignature = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
])

function isPng(filePath) {
  let fd
  try {
    fd = fs.openSync(filePath, 'r')
    const header = Buffer.alloc(pngSignature.length)
    const bytesRead = fs.readSync(fd, header, 0, header.length, 0)
    return bytesRead === header.length && header.equals(pngSignature)
  } catch {
    return false
  } finally {
    if (fd !== undefined) fs.closeSync(fd)
  }
}

export function convertImageToPngFile(inputPath, outputPath, options = {}) {
  const runTool =
    options.runTool ||
    ((command, args) => {
      execFileSync(command, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        timeout: 15_000
      })
    })
  const attempts = [
    ['magick', [`${inputPath}[0]`, outputPath]],
    ['convert', [`${inputPath}[0]`, outputPath]],
    ['ffmpeg', ['-y', '-i', inputPath, '-frames:v', '1', outputPath]],
    ['sips', ['-s', 'format', 'png', inputPath, '--out', outputPath]]
  ]

  for (const [command, args] of attempts) {
    fs.rmSync(outputPath, { force: true })
    try {
      runTool(command, args)
      if (isPng(outputPath)) return true
    } catch {
      // Try the next available converter.
    }
  }

  fs.rmSync(outputPath, { force: true })
  return false
}
