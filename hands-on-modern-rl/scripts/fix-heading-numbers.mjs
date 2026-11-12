/**
 * Write section numbers back into Markdown from the VitePress sidebar (config.mjs), keeping the two in sync.
 *
 * - mismatch: replace the old section number in the body with the sidebar's, and update every heading line
 *   starting with #+ oldMajor.oldMinor (e.g. ### 1.3.1 → ### 1.1.1), so sub-headings don't keep the old number.
 * - missing: rewrite the body's first H1 as `# X.Y` + the sidebar title (matching the text after the number in `text`).
 *
 * Usage:
 *   npm run fix:headings
 *   node scripts/fix-heading-numbers.mjs --dry-run
 * Optional re-check: node scripts/check-heading-numbers.mjs
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..')
const configPath = path.join(repoRoot, 'docs', '.vitepress', 'config.mjs')
const docsRoot = path.join(repoRoot, 'docs')

const dryRun = process.argv.includes('--dry-run')

const pairRe =
  /text:\s*'(\d+)\.(\d+)\s([^']*)'\s*,\s*\r?\n\s*link:\s*'([^']+)'/g

const h1NumRe = /^#\s*(\d+)\.(\d+)\b/
const h1LineRe = /^#\s+/

const src = fs.readFileSync(configPath, 'utf8')

function resolveDocMd(link) {
  const clean = link.replace(/^\//, '')
  if (clean.endsWith('/')) {
    return path.join(docsRoot, clean.replace(/\/$/, ''), 'index.md')
  }
  return path.join(docsRoot, `${clean}.md`)
}

function stripFrontmatterLines(lines) {
  if (lines[0] !== '---') return lines
  const end = lines.indexOf('---', 1)
  if (end === -1) return lines
  return lines.slice(end + 1)
}

function frontmatterSplit(lines) {
  if (lines[0] !== '---') {
    return { before: [], body: lines, afterFmLine: 0 }
  }
  const end = lines.indexOf('---', 1)
  if (end === -1) {
    return { before: lines, body: [], afterFmLine: lines.length }
  }
  return {
    before: lines.slice(0, end + 1),
    body: lines.slice(end + 1),
    afterFmLine: end + 1
  }
}

function firstNumberedH1InBody(bodyLines) {
  const idx = bodyLines.findIndex((line) => h1NumRe.test(line))
  if (idx === -1) return { line: null, index: -1 }
  return { line: bodyLines[idx], index: idx }
}

function firstH1InBody(bodyLines) {
  const idx = bodyLines.findIndex((line) => h1LineRe.test(line))
  if (idx === -1) return { line: null, index: -1 }
  return { line: bodyLines[idx], index: idx }
}

function remapHeadingLine(line, oldM, oldMin, newM, newMin) {
  const re = new RegExp(`^(#{1,6}\\s+)${oldM}\\.${oldMin}(?=\\.|\\s|$)`)
  return line.replace(re, `$1${newM}.${newMin}`)
}

function applyMismatchFix(bodyLines, oldM, oldMin, newM, newMin) {
  return bodyLines.map((line) =>
    remapHeadingLine(line, oldM, oldMin, newM, newMin)
  )
}

function applyMissingFix(bodyLines, newM, newMin, sidebarTitleRest) {
  const h1 = firstH1InBody(bodyLines)
  const newLine = `# ${newM}.${newMin} ${sidebarTitleRest.trim()}`
  if (h1.index === -1) {
    if (
      bodyLines.length === 0 ||
      (bodyLines.length === 1 && bodyLines[0] === '')
    ) {
      return [newLine, '']
    }
    return [newLine, ...bodyLines]
  }
  const next = bodyLines.slice()
  next[h1.index] = newLine
  return next
}

function joinWithEol(lines, eol) {
  if (lines.length === 0) return ''
  return lines.join(eol) + (lines[lines.length - 1] === '' ? '' : eol)
}

const entries = []
let m
while ((m = pairRe.exec(src)) !== null) {
  entries.push({
    major: m[1],
    minor: m[2],
    titleRest: m[3],
    link: m[4].replace(/^\//, '')
  })
}

let changedFiles = 0
let totalEdits = 0

for (const ent of entries) {
  const mdPath = resolveDocMd(ent.link)
  if (!fs.existsSync(mdPath)) {
    console.warn(`[skip] file not found: ${path.relative(repoRoot, mdPath)}`)
    continue
  }

  const raw = fs.readFileSync(mdPath, 'utf8')
  const eol = raw.includes('\r\n') ? '\r\n' : '\n'
  const lines = raw.split(/\r?\n/)
  const { before, body } = frontmatterSplit(lines)

  const numH1 = firstNumberedH1InBody(body)
  const expect = `${ent.major}.${ent.minor}`

  let newBody = null
  let reason = ''

  if (numH1.line) {
    const hm = numH1.line.match(h1NumRe)
    const got = `${hm[1]}.${hm[2]}`
    if (got !== expect) {
      newBody = applyMismatchFix(body, hm[1], hm[2], ent.major, ent.minor)
      reason = `mismatch ${got} → ${expect}`
    }
  } else {
    newBody = applyMissingFix(body, ent.major, ent.minor, ent.titleRest)
    reason = `missing # X.Y, set to ${expect}`
  }

  if (!newBody) continue

  const newRaw = joinWithEol(
    before.length ? [...before, ...newBody] : newBody,
    eol
  )

  if (newRaw !== raw) {
    changedFiles += 1
    totalEdits += 1
    const rel = path.relative(repoRoot, mdPath)
    console.log(`${dryRun ? '[dry-run] ' : ''}${reason}: ${rel}`)
    if (!dryRun) {
      fs.writeFileSync(mdPath, newRaw, 'utf8')
    }
  }
}

console.log(
  `\n${dryRun ? '(not written)' : 'wrote '} ${changedFiles} file(s) (sidebar entries with X.Y: ${entries.length})`
)

// dry-run only prints the plan and exits 0, for local preview
