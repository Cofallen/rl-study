#!/usr/bin/env node

import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { fileURLToPath, pathToFileURL } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const rootDir = path.resolve(__dirname, '..')
const docsDir = path.join(rootDir, 'docs')
const publicDir = path.join(docsDir, 'public')
const packageJsonPath = path.join(rootDir, 'package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const ignoredDirectoryNames = new Set(['.vitepress', 'public', 'node_modules'])

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

function getSiteUrl() {
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`
  }

  if (process.env.SITE_URL) {
    return process.env.SITE_URL.replace(/\/$/, '')
  }

  const { owner, repo } = parseRepository()
  return `https://${owner}.github.io/${repo}`
}

function shouldIgnoreDirectory(name) {
  return ignoredDirectoryNames.has(name) || name.startsWith('_archive')
}

function shouldIncludeMarkdownFile(fullPath, relativePath) {
  const basename = path.basename(relativePath)
  if (!basename.endsWith('.md') || basename.startsWith('_archive')) {
    return false
  }

  return matter(fs.readFileSync(fullPath, 'utf8')).data.sitemap !== false
}

export function scanMarkdownFiles(dir, basePath = '') {
  const files = []
  const entries = fs.readdirSync(dir, { withFileTypes: true })

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    const relativePath = path.join(basePath, entry.name)

    if (entry.isDirectory()) {
      if (shouldIgnoreDirectory(entry.name)) {
        continue
      }
      files.push(...scanMarkdownFiles(fullPath, relativePath))
      continue
    }

    if (entry.isFile() && shouldIncludeMarkdownFile(fullPath, relativePath)) {
      files.push(relativePath)
    }
  }

  return files
}

function toPageUrl(relativePath) {
  const siteUrl = getSiteUrl()
  let pagePath = relativePath.replace(/\\/g, '/').replace(/\.md$/, '')

  if (pagePath === 'index') {
    pagePath = ''
  } else if (pagePath.endsWith('/index')) {
    pagePath = pagePath.slice(0, -6)
  }

  return `${siteUrl}${pagePath ? `/${pagePath}/` : '/'}`
}

function getLastModified(relativePath) {
  const absolutePath = path.join(docsDir, relativePath)
  return fs.statSync(absolutePath).mtime.toISOString()
}

function getPriority(relativePath) {
  if (
    relativePath === 'index.md' ||
    /^(zh|en)\/index\.md$/.test(relativePath)
  ) {
    return '1.0'
  }
  if (
    relativePath.startsWith('guide/') ||
    /^(zh|en)\/guide\//.test(relativePath)
  ) {
    return '0.8'
  }
  return '0.6'
}

function escapeXml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function main() {
  const files = scanMarkdownFiles(docsDir).sort()
  const urls = files.map((relativePath) => ({
    loc: toPageUrl(relativePath),
    lastmod: getLastModified(relativePath),
    priority: getPriority(relativePath)
  }))

  let xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
  xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'

  for (const item of urls) {
    xml += '  <url>\n'
    xml += `    <loc>${escapeXml(item.loc)}</loc>\n`
    xml += `    <lastmod>${item.lastmod}</lastmod>\n`
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += `    <priority>${item.priority}</priority>\n`
    xml += '  </url>\n'
  }

  xml += '</urlset>\n'

  fs.mkdirSync(publicDir, { recursive: true })
  fs.writeFileSync(path.join(publicDir, 'sitemap.xml'), xml)

  console.log(`Generated sitemap for ${urls.length} pages`)
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main()
}
