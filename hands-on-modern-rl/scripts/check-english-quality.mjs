import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const root = process.cwd()
const englishRoot = path.join(root, 'docs', 'en')

function collectMarkdown(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(directory, entry.name)
    if (entry.isDirectory()) return collectMarkdown(absolute)
    if (!entry.isFile() || !entry.name.endsWith('.md')) return []
    if (entry.name.startsWith('_archive')) return []
    return [absolute]
  })
}

function proseOnly(source) {
  const lines = source.split('\n')
  let inFence = false
  const prose = []
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence
      continue
    }
    if (!inFence) prose.push(line.replace(/`[^`]*`/g, ''))
  }
  return prose.join('\n')
}

const ignoredRoutes = new Set()

const checks = [
  {
    label: 'Chinese characters in English prose',
    pattern: /[\u3400-\u9fff]/g
  },
  {
    label: 'full-width Chinese punctuation in English prose',
    pattern: /[（）：，。；]/g
  },
  {
    label: 'nonstandard outcome-reward terminology',
    pattern: /\bresult rewards?\b/gi
  },
  {
    label: 'corrupted reward subscripts',
    pattern: /\bG\*t\b|\br\*\{t\+1\}|\\pi\*|\\phi\*|\bs\*\{?t\b/g
  },
  {
    label: 'truncated Markdown links',
    pattern: /\]\(https\s*(?:$|\|)/gm
  },
  {
    label: 'corrupted section numbers',
    pattern: /\b(?:16\/C|17\/C|3\/C|4\/T\.1)\b/g
  },
  {
    label: 'duplicated English term definitions',
    pattern:
      /\b(?:Re-ranking \(Re-ranking\)|Correctness \(Correctness\)|Isolation \(Isolation\)|Verifiability \(Verifiability\)|Training-Inference Mismatch \(Training-Inference Mismatch\)|preference dataset \(Preference Dataset\))/gi
  },
  {
    label: 'non-idiomatic comparative adjectives',
    pattern: /\bmore dense\b/gi
  }
]

const findings = new Map(checks.map(({ label }) => [label, []]))
for (const file of collectMarkdown(englishRoot)) {
  const route = path.relative(englishRoot, file).split(path.sep).join('/')
  if (ignoredRoutes.has(route)) continue
  const source = proseOnly(fs.readFileSync(file, 'utf8'))
  for (const { label, pattern } of checks) {
    pattern.lastIndex = 0
    const matches = [...source.matchAll(pattern)]
    if (!matches.length) continue
    const lines = matches.map(
      (match) => source.slice(0, match.index).split('\n').length
    )
    findings.get(label).push(`${route}:${[...new Set(lines)].join(',')}`)
  }
}

let failed = false
for (const { label } of checks) {
  const values = findings.get(label)
  console.log(`${label}: ${values.length}`)
  for (const value of values) console.log(`  - ${value}`)
  failed ||= values.length > 0
}

if (failed) process.exitCode = 1
else console.log('English publication-quality checks passed.')
