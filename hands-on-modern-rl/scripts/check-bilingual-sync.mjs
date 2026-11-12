import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const root = process.cwd()
const docsRoot = path.join(root, 'docs')
const configPath = path.join(docsRoot, '.vitepress', 'config.mjs')

function collectMarkdown(directory, prefix = '') {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (entry.name.startsWith('_archive')) return []

    const absolute = path.join(directory, entry.name)
    const relative = path.posix.join(prefix, entry.name)

    if (entry.isDirectory()) return collectMarkdown(absolute, relative)
    if (!entry.isFile() || !entry.name.endsWith('.md')) return []

    return [relative.replace(/\.md$/, '').replace(/\/index$/, '/')]
  })
}

function section(source, start, end) {
  const from = source.indexOf(start)
  const to = source.indexOf(end, from)
  if (from < 0 || to < 0) throw new Error(`Unable to locate ${start}`)
  return source.slice(from, to)
}

function links(source) {
  return [...source.matchAll(/link:\s*'([^']+)'/g)].map((match) => match[1])
}

function entriesWithoutLinks(source) {
  return [...source.matchAll(/text:\s*'([^']+)'\s*\n\s*}/g)].map(
    (match) => match[1]
  )
}

function markdownStructure(file) {
  const source = fs.readFileSync(file, 'utf8')
  const headingLevels = []
  let inFence = false
  for (const line of source.split('\n')) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const heading = line.match(/^(#{1,6})\s+/)
    if (heading) headingLevels.push(heading[1].length)
  }
  return {
    headingLevels,
    codeFences: [...source.matchAll(/^\s*```/gm)].length,
    images: [...source.matchAll(/!\[[^\]]*\]\([^)]*\)|<img\b/g)].length,
    externalUrls: [...source.matchAll(/https?:\/\/[^)\s>]+/g)].map(
      (match) => match[0]
    ),
    linkTargets: [...source.matchAll(/\]\(([^)]+)\)/g)].map((match) => match[1])
  }
}

function sameArray(left, right) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  )
}

const chineseRoutes = collectMarkdown(docsRoot).filter(
  (route) => route !== 'en' && !route.startsWith('en/')
)
const englishRoutes = collectMarkdown(path.join(docsRoot, 'en'))
const chinese = new Set(chineseRoutes)
const english = new Set(englishRoutes)

// These routes intentionally use different structures in the two locales.
// The Chinese home route redirects into the course, while the English home is
// a localized landing page. The Chinese multi-turn route is now a migration
// notice; its English counterpart remains as supplementary legacy reading.
const structuralExceptions = new Set([
  'index',
  'chapter22_agentic/multi-turn-rl'
])

const missingTranslations = chineseRoutes.filter((route) => !english.has(route))
const englishOnly = englishRoutes.filter((route) => !chinese.has(route))
const structuralMismatches = []
for (const route of chineseRoutes) {
  if (!english.has(route)) continue
  if (structuralExceptions.has(route)) continue
  const relative = route.endsWith('/') ? `${route}index.md` : `${route}.md`
  const zh = markdownStructure(path.join(docsRoot, relative))
  const en = markdownStructure(path.join(docsRoot, 'en', relative))
  const fields = []
  if (!sameArray(zh.headingLevels, en.headingLevels)) fields.push('headings')
  if (zh.codeFences !== en.codeFences) fields.push('code fences')
  if (zh.images !== en.images) fields.push('images')
  if (!sameArray(zh.externalUrls, en.externalUrls)) fields.push('external URLs')
  if (zh.linkTargets.length !== en.linkTargets.length) fields.push('link count')
  if (fields.length) structuralMismatches.push(`${route}: ${fields.join(', ')}`)
}

const config = fs.readFileSync(configPath, 'utf8')
const zhSidebar = section(config, 'const zhSidebar', 'const enSidebar')
const enSidebar = section(
  config,
  'const enSidebar',
  'function collectEnglishRoutes'
)
const zhLinks = links(zhSidebar)
const enLinks = links(enSidebar)
const enLinkSet = new Set(enLinks)
const missingNavigationLinks = zhLinks
  .map((route) => `/en${route}`)
  .filter((route) => !enLinkSet.has(route))
const unlinkedEnglishEntries = entriesWithoutLinks(enSidebar)
const brokenEnglishLinks = enLinks.filter((route) => {
  const normalized = route.replace(/^\/en\//, '').replace(/\/$/, '/')
  return route.startsWith('/en/') && !english.has(normalized)
})

const requiredChecks = [
  ['missing translations', missingTranslations],
  ['missing English navigation links', missingNavigationLinks],
  ['English navigation entries without links', unlinkedEnglishEntries],
  ['broken English navigation links', brokenEnglishLinks]
]

let failed = false
for (const [label, values] of requiredChecks) {
  console.log(`${label}: ${values.length}`)
  for (const value of values) console.log(`  - ${value}`)
  failed ||= values.length > 0
}

// Localized pages may legitimately reorganize headings, links, figures, or code
// while preserving the same lesson. Keep structural drift visible for review,
// but do not confuse it with a missing or unreachable translation.
console.log(`structural drift warnings: ${structuralMismatches.length}`)
for (const value of structuralMismatches) console.log(`  - ${value}`)

console.log(`English-only supplementary pages: ${englishOnly.length}`)
for (const value of englishOnly) console.log(`  - ${value}`)
console.log(
  `Documented localized structure exceptions: ${structuralExceptions.size}`
)
for (const value of structuralExceptions) console.log(`  - ${value}`)

if (failed) process.exitCode = 1
else console.log('Bilingual page and navigation coverage are synchronized.')
