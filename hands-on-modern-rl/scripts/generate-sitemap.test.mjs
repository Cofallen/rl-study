import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { scanMarkdownFiles } from './generate-sitemap.mjs'

test('excludes archived and sitemap-disabled Markdown pages', (t) => {
  const docsDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sitemap-test-'))
  t.after(() => fs.rmSync(docsDir, { recursive: true, force: true }))

  fs.mkdirSync(path.join(docsDir, 'guide'), { recursive: true })
  fs.mkdirSync(path.join(docsDir, '_archive_old'), { recursive: true })
  fs.writeFileSync(path.join(docsDir, 'guide', 'index.md'), '# Guide\n')
  fs.writeFileSync(
    path.join(docsDir, '_archive_old', 'index.md'),
    '# Archive\n'
  )
  fs.writeFileSync(path.join(docsDir, '_archive_page.md'), '# Archive\n')
  fs.writeFileSync(
    path.join(docsDir, 'hidden.md'),
    '---\nsitemap: false\n---\n# Hidden\n'
  )

  assert.deepEqual(scanMarkdownFiles(docsDir), ['guide/index.md'])
})
