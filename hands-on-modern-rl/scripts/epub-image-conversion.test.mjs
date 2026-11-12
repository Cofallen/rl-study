import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { convertImageToPngFile } from './epub-image-conversion.mjs'

const pngSignature = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
])

test('tries ImageMagick before ffmpeg and stops after a valid PNG', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'epub-image-test-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const input = path.join(dir, 'input.webp')
  const output = path.join(dir, 'output.png')
  fs.writeFileSync(input, 'webp fixture')
  const commands = []

  const converted = convertImageToPngFile(input, output, {
    runTool(command, args) {
      commands.push(command)
      if (command === 'magick') {
        fs.writeFileSync(args.at(-1), 'not a PNG')
      } else if (command === 'convert') {
        throw new Error('unavailable')
      } else {
        fs.writeFileSync(args.at(-1), pngSignature)
      }
    }
  })

  assert.equal(converted, true)
  assert.deepEqual(commands, ['magick', 'convert', 'ffmpeg'])
})

test('returns false when no tool produces a valid PNG', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'epub-image-test-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const input = path.join(dir, 'input.webp')
  const output = path.join(dir, 'output.png')
  fs.writeFileSync(input, 'webp fixture')
  const commands = []

  const converted = convertImageToPngFile(input, output, {
    runTool(command) {
      commands.push(command)
      throw new Error('unavailable')
    }
  })

  assert.equal(converted, false)
  assert.deepEqual(commands, ['magick', 'convert', 'ffmpeg', 'sips'])
  assert.equal(fs.existsSync(output), false)
})
