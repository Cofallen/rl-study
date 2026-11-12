import { execFileSync, spawnSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const rootDir = path.resolve(path.dirname(__filename), '..')
const docsDir = path.join(rootDir, 'docs')
const outputDir = path.join(docsDir, 'public', 'optimized')
const mermaidSourceDir = path.join(docsDir, 'assets-sources', 'mermaid')
const manifestPath = path.join(outputDir, 'asset-manifest.json')
const mermaidRuntimePath = path.join(
  rootDir,
  'node_modules',
  'mermaid',
  'dist',
  'mermaid.min.js'
)

const rasterExtensions = new Set(['.png', '.jpg', '.jpeg'])
const gifExtensions = new Set(['.gif'])
const svgExtensions = new Set(['.svg'])
const imageExtensions = new Set([
  ...rasterExtensions,
  ...gifExtensions,
  ...svgExtensions
])
const commandAvailability = new Map()

const options = {
  rasterQuality: 82,
  gifQuality: 78,
  maxWidth: 1600,
  minSavingRatio: 0.98
}
const mermaidFontFamily =
  '"Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Arial Unicode MS", sans-serif'
const mermaidRendererVersion = 'cjk-measured-bounds-v3'
const existingManifest = loadExistingManifest()

function loadExistingManifest() {
  if (!fs.existsSync(manifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

function toPosix(value) {
  return value.split(path.sep).join('/')
}

function docsRelative(filePath) {
  return toPosix(path.relative(docsDir, filePath))
}

function publicPathFor(relativePath) {
  return `/optimized/${relativePath}`
}

function fileSize(filePath) {
  return fs.statSync(filePath).size
}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex')
}

function hashFile(filePath) {
  return sha256(fs.readFileSync(filePath))
}

function readGifLoopCount(filePath) {
  const buffer = fs.readFileSync(filePath)
  const marker = Buffer.from('NETSCAPE2.0', 'ascii')
  const markerIndex = buffer.indexOf(marker)

  if (markerIndex < 0) return null

  const loopIndex = markerIndex + marker.length
  if (
    buffer[loopIndex] !== 0x03 ||
    buffer[loopIndex + 1] !== 0x01 ||
    loopIndex + 4 >= buffer.length
  ) {
    return null
  }

  return buffer[loopIndex + 2] + buffer[loopIndex + 3] * 256
}

function readWebpLoopCount(filePath) {
  if (!fs.existsSync(filePath)) return null

  const buffer = fs.readFileSync(filePath)
  const marker = Buffer.from('ANIM', 'ascii')
  const markerIndex = buffer.indexOf(marker)

  if (markerIndex < 0 || markerIndex + 13 >= buffer.length) return null

  return buffer[markerIndex + 12] + buffer[markerIndex + 13] * 256
}

function setWebpLoopCount(filePath, loopCount) {
  if (!commandExists('webpmux')) return

  const loopedPath = `${filePath}.loop.webp`

  try {
    execFileSync(
      'webpmux',
      ['-set', 'loop', String(loopCount), filePath, '-o', loopedPath],
      { stdio: 'ignore' }
    )
    fs.copyFileSync(loopedPath, filePath)
  } finally {
    fs.rmSync(loopedPath, { force: true })
  }
}

function formatSvgNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(3)
}

function normalizeRenderedMermaidSvg(svg, measuredBounds = null) {
  const viewBox = svg
    .match(/\bviewBox=["']([^"']+)["']/i)?.[1]
    ?.trim()
    .split(/[\s,]+/)
    .map(Number)

  if (!viewBox || viewBox.length !== 4) return svg

  const [x, y, width, height] = viewBox
  if (!Number.isFinite(width) || !Number.isFinite(height)) return svg

  const bounds =
    measuredBounds &&
    Number.isFinite(measuredBounds.x) &&
    Number.isFinite(measuredBounds.y) &&
    Number.isFinite(measuredBounds.width) &&
    Number.isFinite(measuredBounds.height) &&
    measuredBounds.width > 0 &&
    measuredBounds.height > 0
      ? measuredBounds
      : { x, y, width, height }

  const minX = Math.floor(bounds.x)
  const minY = Math.floor(bounds.y)
  const maxX = Math.ceil(bounds.x + bounds.width)
  const maxY = Math.ceil(bounds.y + bounds.height)
  const measuredWidth = maxX - minX
  const measuredHeight = maxY - minY

  return svg
    .replace(
      /\bviewBox=["'][^"']*["']/i,
      `viewBox="${formatSvgNumber(minX)} ${formatSvgNumber(minY)} ${formatSvgNumber(measuredWidth)} ${formatSvgNumber(measuredHeight)}"`
    )
    .replace(/\swidth=["'][^"']*["']/i, ` width="${Math.ceil(measuredWidth)}"`)
    .replace(/\sheight=["'][^"']*["']/i, '')
    .replace(/<svg\b/i, `<svg height="${Math.ceil(measuredHeight)}"`)
    .replace(/\sstyle=["']max-width:\s*[^"']*;?["']/i, '')
    .replace(/<svg\b(?![^>]*\boverflow=)/i, '<svg overflow="visible"')
    .replace(
      /<foreignObject\b(?![^>]*\boverflow=)/gi,
      '<foreignObject overflow="visible"'
    )
}

async function measureMermaidSvgBounds(page, svg) {
  return page.evaluate(async (source) => {
    const host = document.createElement('div')
    host.style.cssText =
      'position:absolute;left:0;top:0;display:inline-block;background:#fff;'
    host.innerHTML = source
    document.body.appendChild(host)

    try {
      const svgElement = host.querySelector('svg')
      if (!svgElement) return null

      svgElement.setAttribute('overflow', 'visible')
      svgElement.style.overflow = 'visible'
      svgElement
        .querySelectorAll('foreignObject')
        .forEach((element) => element.setAttribute('overflow', 'visible'))

      if (document.fonts?.ready) await document.fonts.ready

      const viewBox = svgElement.viewBox.baseVal
      const svgRect = svgElement.getBoundingClientRect()
      if (
        !svgRect.width ||
        !svgRect.height ||
        !viewBox.width ||
        !viewBox.height
      ) {
        return null
      }

      const ignored = new Set([
        'defs',
        'desc',
        'filter',
        'linearGradient',
        'marker',
        'metadata',
        'radialGradient',
        'script',
        'style',
        'title'
      ])
      const rects = []

      for (const element of svgElement.querySelectorAll('*')) {
        if (ignored.has(element.tagName)) continue

        const style = window.getComputedStyle(element)
        if (style.display === 'none' || style.visibility === 'hidden') continue

        const rect = element.getBoundingClientRect()
        if (rect.width > 0 && rect.height > 0) {
          rects.push(rect)
        }
      }

      if (!rects.length) return null

      const left = Math.min(...rects.map((rect) => rect.left))
      const top = Math.min(...rects.map((rect) => rect.top))
      const right = Math.max(...rects.map((rect) => rect.right))
      const bottom = Math.max(...rects.map((rect) => rect.bottom))
      const scaleX = viewBox.width / svgRect.width
      const scaleY = viewBox.height / svgRect.height

      return {
        x: viewBox.x + (left - svgRect.left) * scaleX,
        y: viewBox.y + (top - svgRect.top) * scaleY,
        width: (right - left) * scaleX,
        height: (bottom - top) * scaleY
      }
    } finally {
      host.remove()
    }
  }, svg)
}

function commandExists(command) {
  if (commandAvailability.has(command)) {
    return commandAvailability.get(command)
  }

  const result = spawnSync('command', ['-v', command], {
    shell: true,
    stdio: 'ignore'
  })
  const exists = result.status === 0
  commandAvailability.set(command, exists)
  return exists
}

function commandPath(command) {
  const result = spawnSync('command', ['-v', command], {
    shell: true,
    encoding: 'utf8'
  })

  if (result.status !== 0) return null
  return result.stdout.trim().split('\n')[0] || null
}

function findBrowserExecutable() {
  const envCandidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    process.env.CHROME_PATH
  ].filter(Boolean)
  const macCandidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
  ]
  const pathCandidates = [
    'chromium',
    'chromium-browser',
    'google-chrome',
    'google-chrome-stable',
    'chrome',
    'msedge'
  ]
    .map(commandPath)
    .filter(Boolean)

  return [...envCandidates, ...macCandidates, ...pathCandidates].find(
    (candidate) => fs.existsSync(candidate)
  )
}

async function createBrowser() {
  // Chromium cannot register its required Mach services from Codex's macOS
  // seatbelt sandbox. Attempting to launch it there aborts the process and
  // makes macOS show a misleading “Google Chrome quit unexpectedly” dialog.
  // An explicitly approved, unsandboxed build does not set CODEX_SANDBOX and
  // can still render SVG and Mermaid assets normally.
  if (
    process.platform === 'darwin' &&
    process.env.CODEX_SANDBOX === 'seatbelt'
  ) {
    return {
      error:
        'browser rendering skipped inside the Codex macOS sandbox; rerun the build with unsandboxed approval to refresh SVG and Mermaid assets'
    }
  }

  const executablePath = findBrowserExecutable()

  if (!executablePath) {
    return { error: 'missing Chrome or Chromium executable' }
  }

  let puppeteer
  try {
    puppeteer = await import('puppeteer-core')
  } catch (error) {
    return { error: `missing puppeteer-core: ${error.message}` }
  }

  try {
    const browser = await puppeteer.default.launch({
      executablePath,
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    })

    return { browser, executablePath }
  } catch (error) {
    return { error: `browser launch failed: ${error.message}` }
  }
}

function walk(dir, files = []) {
  if (!fs.existsSync(dir)) return files

  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name)
    const relativePath = docsRelative(fullPath)

    if (entry.isDirectory()) {
      if (
        relativePath === '.vitepress' ||
        relativePath.startsWith('.vitepress/') ||
        relativePath === 'public' ||
        relativePath.startsWith('public/') ||
        relativePath === 'public/optimized' ||
        relativePath.startsWith('public/optimized/') ||
        relativePath === 'assets-sources' ||
        relativePath.startsWith('assets-sources/')
      ) {
        continue
      }

      walk(fullPath, files)
      continue
    }

    if (!entry.isFile()) continue
    files.push(fullPath)
  }

  return files
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
}

function outputRelativeForImage(sourceRelative, extension = '.webp') {
  return `${sourceRelative.replace(/\.[^.]+$/, '')}${extension}`
}

function optimizeRaster(sourcePath, sourceRelative) {
  const outputRelative = outputRelativeForImage(sourceRelative)
  const outputPath = path.join(outputDir, outputRelative)
  const cached = useCachedOutput(sourcePath, outputPath, outputRelative)

  if (cached) return cached

  if (!commandExists('cwebp')) {
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      'missing cwebp'
    )
  }

  const tempPath = path.join(
    os.tmpdir(),
    `homrl-${sha256(sourcePath).slice(0, 12)}.webp`
  )

  const args = [
    '-quiet',
    '-q',
    String(options.rasterQuality),
    '-m',
    '6',
    '-metadata',
    'none',
    '-resize',
    String(options.maxWidth),
    '0',
    '-resize_mode',
    'down_only',
    sourcePath,
    '-o',
    tempPath
  ]

  try {
    execFileSync('cwebp', args, { stdio: 'ignore' })
  } catch (error) {
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      `cwebp failed: ${error.message}`
    )
  }

  return keepIfSmaller(sourcePath, tempPath, outputPath, outputRelative)
}

function optimizeGif(sourcePath, sourceRelative) {
  const outputRelative = outputRelativeForImage(sourceRelative)
  const outputPath = path.join(outputDir, outputRelative)
  const sourceLoopCount = readGifLoopCount(sourcePath)
  const cached = useCachedOutput(sourcePath, outputPath, outputRelative)

  if (cached) {
    const cachedOutputPath = cached.optimizedPath
      ? path.join(docsDir, cached.optimizedPath)
      : outputPath
    const cachedLoopCount = readWebpLoopCount(cachedOutputPath)

    if (sourceLoopCount === null || cachedLoopCount === sourceLoopCount) {
      if (sourceLoopCount !== null) cached.loopCount = sourceLoopCount
      return cached
    }
  }

  if (!commandExists('gif2webp')) {
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      'missing gif2webp'
    )
  }

  const tempPath = path.join(
    os.tmpdir(),
    `homrl-${sha256(sourcePath).slice(0, 12)}.webp`
  )

  const args = [
    '-quiet',
    '-mixed',
    '-q',
    String(options.gifQuality),
    '-metadata',
    'none',
    sourcePath,
    '-o',
    tempPath
  ]

  try {
    execFileSync('gif2webp', args, { stdio: 'ignore' })
    if (sourceLoopCount !== null) {
      setWebpLoopCount(tempPath, sourceLoopCount)
    }
  } catch (error) {
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      `gif2webp failed: ${error.message}`
    )
  }

  const result = keepIfSmaller(sourcePath, tempPath, outputPath, outputRelative)
  if (sourceLoopCount !== null) result.loopCount = sourceLoopCount
  return result
}

function optimizeSvg(sourcePath, sourceRelative) {
  const outputRelative = sourceRelative
  const outputPath = path.join(outputDir, outputRelative)
  const source = fs.readFileSync(sourcePath, 'utf8')
  const optimized = `${source
    .replace(/^<\?xml[\s\S]*?\?>\s*/i, '')
    .replace(/^<!DOCTYPE[\s\S]*?>\s*/i, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/>\s+</g, '><')
    .trim()}\n`
  const tempPath = path.join(
    os.tmpdir(),
    `homrl-${sha256(sourcePath).slice(0, 12)}.svg`
  )

  fs.writeFileSync(tempPath, optimized)
  return keepIfSmaller(sourcePath, tempPath, outputPath, outputRelative)
}

async function captureSvgAsWebp(page, sourcePath, outputPath) {
  const source = fs.readFileSync(sourcePath)
  const dataUrl = `data:image/svg+xml;base64,${source.toString('base64')}`
  const html = `<style>body{margin:0;background:#fff}img{display:block;max-width:${options.maxWidth}px;height:auto;background:#fff}</style><img id="target" src="${dataUrl}">`

  await page.setViewport({
    width: options.maxWidth,
    height: options.maxWidth,
    deviceScaleFactor: 1
  })
  await page.setContent(html)
  await page.waitForSelector('#target')

  const box = await page.$eval('#target', (image) => {
    const rect = image.getBoundingClientRect()
    return {
      width: Math.ceil(rect.width),
      height: Math.ceil(rect.height)
    }
  })

  if (!box.width || !box.height) {
    throw new Error('SVG rendered with zero dimensions')
  }

  await page.setViewport({
    width: Math.max(1, box.width),
    height: Math.max(1, box.height),
    deviceScaleFactor: 1
  })

  const element = await page.$('#target')
  await element.screenshot({
    path: outputPath,
    type: 'webp',
    quality: options.rasterQuality,
    omitBackground: false
  })
}

async function renderSvgRasters(manifest, sharedBrowser = null) {
  const svgAssets = Object.entries(manifest.assets).filter(
    ([, asset]) => asset.type === 'svg'
  )

  if (!svgAssets.length) {
    manifest.svgRasterizer = { status: 'skipped', reason: 'no SVG assets' }
    return
  }

  const pendingAssets = []
  let rasterized = 0

  for (const [sourceRelative, asset] of svgAssets) {
    const outputRelative = outputRelativeForImage(sourceRelative)
    const outputPath = path.join(outputDir, outputRelative)
    const previous = existingManifest.assets?.[sourceRelative]

    if (
      previous?.sourceHash === asset.sourceHash &&
      previous.rasterized === true &&
      previous.status === 'optimized' &&
      fs.existsSync(outputPath)
    ) {
      if (asset.optimizedPath) {
        fs.rmSync(path.join(docsDir, asset.optimizedPath), { force: true })
      }

      Object.assign(asset, {
        status: 'optimized',
        optimized: publicPathFor(outputRelative),
        optimizedPath: docsRelative(outputPath),
        optimizedBytes: fileSize(outputPath),
        savingRatio: Number(
          (fileSize(outputPath) / asset.sourceBytes).toFixed(4)
        ),
        rasterized: true,
        preserved: true
      })
      manifest.svgSources[sourceRelative].status = asset.status
      manifest.svgSources[sourceRelative].optimized = asset.optimized
      manifest.svgSources[sourceRelative].rasterized = true
      rasterized += 1
      continue
    }

    if (
      previous?.sourceHash === asset.sourceHash &&
      previous.rasterizeStatus === 'not-smaller'
    ) {
      Object.assign(asset, {
        rasterizeStatus: previous.rasterizeStatus,
        rasterizeReason: previous.rasterizeReason
      })
      manifest.svgSources[sourceRelative].rasterizeStatus =
        previous.rasterizeStatus
      manifest.svgSources[sourceRelative].rasterizeReason =
        previous.rasterizeReason
      continue
    }

    pendingAssets.push([sourceRelative, asset])
  }

  if (!pendingAssets.length) {
    manifest.svgRasterizer = {
      status: 'optimized',
      rasterized,
      total: svgAssets.length,
      preserved: true
    }
    return
  }

  let browser = sharedBrowser
  let ownBrowser = false

  if (!browser) {
    const result = await createBrowser()
    if (result.error) {
      manifest.svgRasterizer = {
        status: 'skipped',
        reason: result.error,
        rasterized,
        total: svgAssets.length
      }
      return
    }
    browser = result.browser
    ownBrowser = true
  }

  try {
    const page = await browser.newPage()

    for (const [sourceRelative, asset] of pendingAssets) {
      const sourcePath = path.join(docsDir, sourceRelative)
      const outputRelative = outputRelativeForImage(sourceRelative)
      const outputPath = path.join(outputDir, outputRelative)
      const tempPath = path.join(
        os.tmpdir(),
        `homrl-${sha256(sourcePath).slice(0, 12)}.webp`
      )

      try {
        await captureSvgAsWebp(page, sourcePath, tempPath)

        if (!fs.existsSync(tempPath)) {
          throw new Error('SVG rasterizer did not produce an output file')
        }

        const currentBestBytes = asset.optimizedBytes || asset.sourceBytes
        const rasterBytes = fileSize(tempPath)

        if (
          rasterBytes < Math.round(currentBestBytes * options.minSavingRatio)
        ) {
          ensureDir(outputPath)
          fs.copyFileSync(tempPath, outputPath)

          if (asset.optimizedPath) {
            fs.rmSync(path.join(docsDir, asset.optimizedPath), { force: true })
          }

          Object.assign(asset, {
            status: 'optimized',
            optimized: publicPathFor(outputRelative),
            optimizedPath: docsRelative(outputPath),
            optimizedBytes: rasterBytes,
            savingRatio: Number((rasterBytes / asset.sourceBytes).toFixed(4)),
            rasterized: true
          })
          manifest.svgSources[sourceRelative].status = asset.status
          manifest.svgSources[sourceRelative].optimized = asset.optimized
          manifest.svgSources[sourceRelative].rasterized = true
          rasterized += 1
        } else {
          Object.assign(asset, {
            rasterizeStatus: 'not-smaller',
            rasterizeReason: 'rasterized WebP was not smaller'
          })
          manifest.svgSources[sourceRelative].rasterizeStatus =
            asset.rasterizeStatus
          manifest.svgSources[sourceRelative].rasterizeReason =
            asset.rasterizeReason
        }

        fs.rmSync(tempPath, { force: true })
      } catch (error) {
        Object.assign(asset, {
          rasterizeStatus: 'failed',
          rasterizeReason: error.message
        })
        manifest.svgSources[sourceRelative].rasterizeStatus =
          asset.rasterizeStatus
        manifest.svgSources[sourceRelative].rasterizeReason = error.message
      }
    }

    manifest.svgRasterizer = {
      status: 'optimized',
      rasterized,
      total: svgAssets.length,
      executablePath: browser.process()?.spawnfile || null
    }
  } finally {
    if (ownBrowser && browser) await browser.close()
  }
}

function keepIfSmaller(sourcePath, tempPath, outputPath, outputRelative) {
  const sourceBytes = fileSize(sourcePath)

  if (!fs.existsSync(tempPath)) {
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      'optimization command did not produce an output file'
    )
  }

  const optimizedBytes = fileSize(tempPath)

  if (optimizedBytes >= Math.round(sourceBytes * options.minSavingRatio)) {
    fs.rmSync(tempPath, { force: true })
    return useExistingOutput(
      sourcePath,
      outputPath,
      outputRelative,
      'optimized file was not smaller'
    )
  }

  ensureDir(outputPath)
  fs.copyFileSync(tempPath, outputPath)
  fs.rmSync(tempPath, { force: true })

  return {
    status: 'optimized',
    optimized: publicPathFor(outputRelative),
    optimizedPath: docsRelative(outputPath),
    sourceBytes,
    optimizedBytes,
    savingRatio: Number((optimizedBytes / sourceBytes).toFixed(4))
  }
}

function useExistingOutput(sourcePath, outputPath, outputRelative, reason) {
  const sourceBytes = fileSize(sourcePath)

  if (!fs.existsSync(outputPath)) {
    return { status: 'skipped', reason, sourceBytes }
  }

  const sourceRelative = docsRelative(sourcePath)
  const sourceHash = hashFile(sourcePath)
  const previousHash = existingManifest.assets?.[sourceRelative]?.sourceHash
  const optimizedBytes = fileSize(outputPath)

  if (previousHash && previousHash !== sourceHash) {
    return {
      status: 'skipped',
      reason: `${reason}; existing optimized file may be stale`,
      sourceBytes,
      optimizedBytes
    }
  }

  if (optimizedBytes >= Math.round(sourceBytes * options.minSavingRatio)) {
    return {
      status: 'skipped',
      reason,
      sourceBytes,
      optimizedBytes
    }
  }

  return {
    status: 'optimized',
    optimized: publicPathFor(outputRelative),
    optimizedPath: docsRelative(outputPath),
    sourceBytes,
    optimizedBytes,
    savingRatio: Number((optimizedBytes / sourceBytes).toFixed(4)),
    preserved: true,
    reason
  }
}

function useCachedOutput(sourcePath, outputPath, outputRelative) {
  const sourceRelative = docsRelative(sourcePath)
  const previous = existingManifest.assets?.[sourceRelative]

  if (!previous || previous.sourceHash !== hashFile(sourcePath)) return null

  const previousOutputPath = previous.optimizedPath
    ? path.join(docsDir, previous.optimizedPath)
    : outputPath

  if (
    previous.status === 'optimized' &&
    previous.optimizedPath &&
    fs.existsSync(previousOutputPath)
  ) {
    const sourceBytes = fileSize(sourcePath)
    const optimizedBytes = fileSize(previousOutputPath)

    return {
      status: 'optimized',
      optimized: previous.optimized || publicPathFor(outputRelative),
      optimizedPath: docsRelative(previousOutputPath),
      sourceBytes,
      optimizedBytes,
      savingRatio: Number((optimizedBytes / sourceBytes).toFixed(4)),
      preserved: true
    }
  }

  if (
    previous.status === 'skipped' &&
    previous.reason === 'optimized file was not smaller'
  ) {
    return {
      status: 'skipped',
      reason: previous.reason,
      sourceBytes: fileSize(sourcePath),
      optimizedBytes: previous.optimizedBytes
    }
  }

  return null
}

function collectImages() {
  return walk(docsDir).filter((filePath) =>
    imageExtensions.has(path.extname(filePath).toLowerCase())
  )
}

function collectMarkdownFiles() {
  return walk(docsDir).filter((filePath) => filePath.endsWith('.md'))
}

function extractMermaidBlocks(markdownPath) {
  const source = fs.readFileSync(markdownPath, 'utf8')
  const blocks = []
  const pattern = /^```mermaid[^\n]*\n([\s\S]*?)^```/gm
  let match
  let index = 0

  while ((match = pattern.exec(source)) !== null) {
    index += 1
    const body = match[1].trimEnd() + '\n'
    const pageRelative = docsRelative(markdownPath)
    const pageBase = pageRelative.replace(/\.md$/, '')
    const blockId = `${pageBase.replace(/\/index$/, '').replace(/\//g, '__')}__${String(index).padStart(2, '0')}`
    const sourceRelative = `${pageBase}-${String(index).padStart(2, '0')}.mmd`
    const sourcePath = path.join(mermaidSourceDir, sourceRelative)

    ensureDir(sourcePath)
    fs.writeFileSync(sourcePath, body)

    blocks.push({
      id: blockId,
      page: pageRelative,
      index,
      source: docsRelative(sourcePath),
      hash: sha256(body).slice(0, 16),
      status: 'source-only'
    })
  }

  return blocks
}

function outputRelativeForMermaid(block) {
  return `mermaid/${block.source
    .replace(/^assets-sources\/mermaid\//, '')
    .replace(/\.mmd$/, '.svg')}`
}

function buildExistingMermaidMap() {
  return new Map(
    (existingManifest.mermaid || []).map((block) => [block.id, block])
  )
}

async function renderMermaidBlocks(manifest, sharedBrowser = null) {
  const blocks = manifest.mermaid
  const existingMermaid = buildExistingMermaidMap()

  if (!blocks.length) {
    manifest.mermaidRenderer = { status: 'skipped', reason: 'no blocks' }
    return
  }

  let rendered = 0
  const pendingBlocks = []

  for (const block of blocks) {
    const outputRelative = outputRelativeForMermaid(block)
    const outputPath = path.join(outputDir, outputRelative)
    const previous = existingMermaid.get(block.id)

    if (
      previous?.hash === block.hash &&
      previous.rendererVersion === mermaidRendererVersion &&
      previous.status === 'optimized' &&
      fs.existsSync(outputPath)
    ) {
      Object.assign(block, {
        status: 'optimized',
        type: 'mermaid-svg',
        rendererVersion: previous.rendererVersion,
        optimized: previous.optimized,
        optimizedPath: previous.optimizedPath,
        sourceBytes: previous.sourceBytes,
        optimizedBytes: fileSize(outputPath),
        preserved: true
      })
      rendered += 1
      continue
    }

    pendingBlocks.push(block)
  }

  if (!pendingBlocks.length) {
    manifest.mermaidRenderer = {
      status: 'optimized',
      rendered,
      total: blocks.length,
      preserved: true
    }
    return
  }

  if (process.env.SKIP_MERMAID_RENDER === '1') {
    manifest.mermaidRenderer = {
      status: 'skipped',
      reason: 'SKIP_MERMAID_RENDER=1'
    }
    return
  }

  let browser = sharedBrowser
  let ownBrowser = false

  if (!browser) {
    const result = await createBrowser()
    if (result.error) {
      manifest.mermaidRenderer = {
        status: 'skipped',
        reason: result.error
      }
      return
    }
    browser = result.browser
    ownBrowser = true
  }

  if (!fs.existsSync(mermaidRuntimePath)) {
    manifest.mermaidRenderer = {
      status: 'skipped',
      reason: 'missing mermaid runtime'
    }
    return
  }

  try {
    const page = await browser.newPage()
    await page.setViewport({
      width: options.maxWidth,
      height: options.maxWidth,
      deviceScaleFactor: 1
    })
    await page.setContent(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body {
        margin: 0;
        background: #fff;
        font-family: ${mermaidFontFamily};
      }
    </style>
  </head>
  <body></body>
</html>`)
    await page.evaluateHandle('document.fonts && document.fonts.ready')
    await page.addScriptTag({ path: mermaidRuntimePath })
    await page.evaluate((fontFamily) => {
      window.mermaid.initialize({
        securityLevel: 'loose',
        startOnLoad: false,
        theme: 'default',
        themeVariables: {
          fontFamily,
          fontSize: '16px'
        },
        flowchart: { htmlLabels: true, useMaxWidth: false },
        sequence: { useMaxWidth: false },
        gantt: { useMaxWidth: false }
      })
    }, mermaidFontFamily)

    for (const block of pendingBlocks) {
      const outputRelative = outputRelativeForMermaid(block)
      const outputPath = path.join(outputDir, outputRelative)
      const sourcePath = path.join(docsDir, block.source)
      const source = fs.readFileSync(sourcePath, 'utf8')

      try {
        const svg = await page.evaluate(
          async ({ id, graph }) => {
            const result = await window.mermaid.render(id, graph)
            return result.svg
          },
          {
            id: `mermaid-${block.id}`,
            graph: source
          }
        )
        const measuredBounds = await measureMermaidSvgBounds(page, svg)

        ensureDir(outputPath)
        fs.writeFileSync(
          outputPath,
          `${normalizeRenderedMermaidSvg(svg, measuredBounds).trim()}\n`
        )
        Object.assign(block, {
          status: 'optimized',
          type: 'mermaid-svg',
          rendererVersion: mermaidRendererVersion,
          optimized: publicPathFor(outputRelative),
          optimizedPath: docsRelative(outputPath),
          sourceBytes: Buffer.byteLength(source),
          optimizedBytes: fileSize(outputPath)
        })
        rendered += 1
      } catch (error) {
        Object.assign(block, {
          status: 'source-only',
          reason: `Mermaid render failed: ${error.message}`
        })
      }
    }

    manifest.mermaidRenderer = {
      status: 'optimized',
      rendered,
      total: blocks.length,
      executablePath
    }
  } catch (error) {
    manifest.mermaidRenderer = {
      status: 'skipped',
      reason: `browser launch failed: ${error.message}`
    }
  } finally {
    if (ownBrowser && browser) await browser.close()
  }
}

function writeAssetsReadme(manifest) {
  const readmePath = path.join(docsDir, 'assets-sources', 'README.txt')
  const optimizedCount = Object.values(manifest.assets).filter(
    (asset) => asset.status === 'optimized'
  ).length
  const svgCount = Object.keys(manifest.svgSources).length
  const mermaidRenderedCount = manifest.mermaid.filter(
    (block) => block.status === 'optimized'
  ).length

  const body = `# Asset Sources

This directory stores generated source-side records for course media.

- Original raster and SVG files remain in their chapter folders under \`docs/\`.
- Optimized files for GitHub Pages live under \`docs/public/optimized/\`.
- \`docs/public/optimized/asset-manifest.json\` maps each source file to its optimized derivative.
- Mermaid blocks are extracted to \`docs/assets-sources/mermaid/\` for lookup and offline rendering. Edit the original Markdown block when changing a diagram.
- Rendered Mermaid SVG files live under \`docs/public/optimized/mermaid/\` when Chrome or Chromium is available.

Current manifest summary:

- Optimized image derivatives: ${optimizedCount}
- SVG source records: ${svgCount}
- Mermaid source records: ${manifest.mermaid.length}
- Rendered Mermaid SVG derivatives: ${mermaidRenderedCount}

Run \`npm run assets:optimize\` after adding or changing course images.
`

  fs.mkdirSync(path.dirname(readmePath), { recursive: true })
  fs.writeFileSync(readmePath, body)
}

async function main() {
  fs.mkdirSync(outputDir, { recursive: true })

  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    options,
    assets: {},
    svgSources: {},
    mermaid: []
  }

  for (const imagePath of collectImages()) {
    const relativePath = docsRelative(imagePath)
    const extension = path.extname(imagePath).toLowerCase()
    const sourceBytes = fileSize(imagePath)
    const source = `docs/${relativePath}`
    const sourceHash = hashFile(imagePath)

    if (rasterExtensions.has(extension)) {
      manifest.assets[relativePath] = {
        source,
        sourceHash,
        sourceBytes,
        type: extension.slice(1),
        ...optimizeRaster(imagePath, relativePath)
      }
      continue
    }

    if (gifExtensions.has(extension)) {
      manifest.assets[relativePath] = {
        source,
        sourceHash,
        sourceBytes,
        type: 'gif',
        animated: true,
        ...optimizeGif(imagePath, relativePath)
      }
      continue
    }

    if (svgExtensions.has(extension)) {
      const optimizedSvg = optimizeSvg(imagePath, relativePath)

      manifest.assets[relativePath] = {
        source,
        sourceHash,
        sourceBytes,
        type: 'svg',
        ...optimizedSvg
      }
      manifest.svgSources[relativePath] = {
        source,
        sourceHash,
        sourceBytes,
        type: 'svg',
        status: optimizedSvg.status,
        optimized: optimizedSvg.optimized,
        note: 'Editable SVG source is kept in place. The optimized derivative is a WebP raster when Chrome or Chromium is available and the raster is smaller; otherwise it falls back to a minified SVG.'
      }
    }
  }

  const browserResult = await createBrowser()
  const sharedBrowser = browserResult.error ? null : browserResult.browser

  try {
    await renderSvgRasters(manifest, sharedBrowser)

    for (const markdownPath of collectMarkdownFiles()) {
      manifest.mermaid.push(...extractMermaidBlocks(markdownPath))
    }

    await renderMermaidBlocks(manifest, sharedBrowser)
  } finally {
    if (sharedBrowser) await sharedBrowser.close()
  }

  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
  writeAssetsReadme(manifest)

  const optimizedCount = Object.values(manifest.assets).filter(
    (asset) => asset.status === 'optimized'
  ).length
  const skippedCount = Object.values(manifest.assets).length - optimizedCount

  console.log(
    `Optimized ${optimizedCount} images, skipped ${skippedCount}, indexed ${
      Object.keys(manifest.svgSources).length
    } SVGs and ${manifest.mermaid.length} Mermaid blocks${
      manifest.svgRasterizer?.rasterized
        ? `, rasterized ${manifest.svgRasterizer.rasterized} SVGs`
        : ''
    }${
      manifest.mermaidRenderer?.rendered
        ? `, rendered ${manifest.mermaidRenderer.rendered} Mermaid SVGs`
        : ''
    }.`
  )
}

await main()
