import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

const scriptsDir = path.dirname(fileURLToPath(import.meta.url))
const rootDir = path.resolve(scriptsDir, '..')
const distDir = path.join(rootDir, 'docs', '.vitepress', 'dist')
const fileName = `epub-inline-math-test-${process.pid}.epub`
const epubPath = path.join(distDir, fileName)

test(
  'renders inline math at the end of a list item as MathML',
  { timeout: 900_000 },
  () => {
    try {
      execFileSync(
        process.execPath,
        [path.join(scriptsDir, 'build-epub.mjs')],
        {
          cwd: rootDir,
          env: { ...process.env, EPUB_FILE_NAME: fileName },
          stdio: 'pipe',
          timeout: 900_000
        }
      )

      const entries = execFileSync('zipinfo', ['-1', epubPath], {
        encoding: 'utf8'
      })
        .trim()
        .split('\n')
        .filter((entry) => entry.endsWith('.xhtml'))

      const phrase = '给定状态直接输出动作'
      let targetLine = ''
      for (const entry of entries) {
        const xhtml = execFileSync('unzip', ['-p', epubPath, entry], {
          encoding: 'utf8',
          maxBuffer: 10 * 1024 * 1024
        })
        targetLine =
          xhtml.split('\n').find((line) => line.includes(phrase)) || ''
        if (targetLine) break
      }

      assert.ok(targetLine, `expected an XHTML line containing: ${phrase}`)
      assert.doesNotMatch(targetLine, /\$a = \\pi\(s\)\$/)
      assert.equal((targetLine.match(/<math\b/g) || []).length, 2)
      assert.match(targetLine, /<\/math><\/li>$/)
    } finally {
      fs.rmSync(epubPath, { force: true })
    }
  }
)
