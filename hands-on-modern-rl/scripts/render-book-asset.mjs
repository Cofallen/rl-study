import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const rootDir = path.resolve(path.dirname(__filename), '..')
const mermaidFontFamily =
  '"Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Arial Unicode MS", sans-serif'

function findBrowserExecutable() {
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser'
  ].filter(Boolean)

  return candidates.find((candidate) => fs.existsSync(candidate)) || null
}

function sanitizeInlineSvg(source) {
  return source
    .replace(/^\s*<\?xml[\s\S]*?\?>\s*/i, '')
    .replace(/^\s*<!doctype[\s\S]*?>\s*/i, '')
}

function numericLength(value) {
  const match = String(value || '').match(/^([0-9.]+)(?:px|pt|mm|cm|in)?$/i)
  if (!match) return null
  const number = Number(match[1])
  return Number.isFinite(number) && number > 0 ? number : null
}

function extractSvgSize(svg) {
  const tag = svg.match(/<svg\b([^>]*)>/i)?.[1] || ''
  const width = numericLength(tag.match(/\bwidth=["']([^"']+)["']/i)?.[1])
  const height = numericLength(tag.match(/\bheight=["']([^"']+)["']/i)?.[1])
  const viewBox = tag
    .match(/\bviewBox=["']([^"']+)["']/i)?.[1]
    ?.trim()
    .split(/[\s,]+/)
    .map(Number)

  const viewBoxWidth =
    viewBox?.length === 4 && Number.isFinite(viewBox[2]) ? viewBox[2] : null
  const viewBoxHeight =
    viewBox?.length === 4 && Number.isFinite(viewBox[3]) ? viewBox[3] : null

  const rawWidth = width || viewBoxWidth || 960
  const rawHeight = height || viewBoxHeight || Math.round((rawWidth * 9) / 16)
  const scale = rawWidth > 1280 ? 1280 / rawWidth : 1

  return {
    width: Math.max(320, Math.round(rawWidth * scale)),
    height: Math.max(180, Math.round(rawHeight * scale))
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

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

async function withBrowser(callback) {
  const executablePath = findBrowserExecutable()
  if (!executablePath) {
    throw new Error(
      'No Chrome/Chromium executable found. Set PUPPETEER_EXECUTABLE_PATH.'
    )
  }

  const puppeteerModule = await import('puppeteer-core')
  const puppeteer = puppeteerModule.default || puppeteerModule
  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage'
    ]
  })

  try {
    return await callback(browser)
  } finally {
    await browser.close()
  }
}

async function captureElement(browser, html, outputPath) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true })

  const page = await browser.newPage()
  await page.setViewport({
    width: 1200,
    height: 800,
    deviceScaleFactor: 2
  })
  await page.setContent(html, { waitUntil: 'load' })
  await page.evaluateHandle('document.fonts && document.fonts.ready')

  const box = await page.$eval('#capture', (element) => {
    const rect = element.getBoundingClientRect()
    return {
      width: Math.ceil(rect.width),
      height: Math.ceil(rect.height)
    }
  })

  await page.setViewport({
    width: Math.min(Math.max(box.width + 24, 320), 2400),
    height: Math.min(Math.max(box.height + 24, 180), 3200),
    deviceScaleFactor: 2
  })

  const element = await page.$('#capture')
  if (!element) throw new Error('Rendered element was not found.')
  await element.screenshot({ path: outputPath, omitBackground: false })
  await page.close()
}

async function renderSvg(inputPath, outputPath) {
  const svg = sanitizeInlineSvg(fs.readFileSync(inputPath, 'utf8'))
  const size = extractSvgSize(svg)
  const baseHref = pathToFileURL(`${path.dirname(inputPath)}${path.sep}`).href
  const html = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <base href="${baseHref}">
    <style>
      body { margin: 0; background: #fff; }
      #capture {
        display: inline-block;
        padding: 8px;
        background: #fff;
      }
      #capture svg {
        width: ${size.width}px;
        height: auto;
        max-width: none;
      }
    </style>
  </head>
  <body>
    <div id="capture">${svg}</div>
  </body>
</html>`

  await withBrowser((browser) => captureElement(browser, html, outputPath))
}

async function renderMermaid(inputPath, outputPath) {
  const diagram = fs.readFileSync(inputPath, 'utf8')
  const mermaidPath = path.join(
    rootDir,
    'node_modules',
    'mermaid',
    'dist',
    'mermaid.min.js'
  )
  if (!fs.existsSync(mermaidPath)) {
    throw new Error('Mermaid bundle not found. Run npm install first.')
  }

  await withBrowser(async (browser) => {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true })

    const page = await browser.newPage()
    await page.setViewport({
      width: 1400,
      height: 900,
      deviceScaleFactor: 2
    })
    await page.setContent(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body {
        margin: 0;
        background: #fff;
        color: #111;
        font-family: ${mermaidFontFamily};
      }
      #capture {
        display: inline-block;
        padding: 12px;
        background: #fff;
      }
      svg {
        width: auto;
        height: auto;
        max-width: none;
        overflow: visible;
      }
    </style>
  </head>
  <body>
    <div id="capture"></div>
  </body>
</html>`)
    await page.addScriptTag({ path: mermaidPath })

    const renderResult = await page.evaluate(
      async ({ source, fontFamily }) => {
        try {
          window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'loose',
            theme: 'default',
            themeVariables: {
              fontFamily,
              fontSize: '16px'
            },
            flowchart: { htmlLabels: true, useMaxWidth: false },
            sequence: { useMaxWidth: false },
            gantt: { useMaxWidth: false }
          })
          const result = await window.mermaid.render(
            `diagram-${Date.now()}`,
            source
          )
          return { svg: result.svg }
        } catch (error) {
          return { error: error?.message || String(error) }
        }
      },
      { source: diagram, fontFamily: mermaidFontFamily }
    )

    if (renderResult.error) throw new Error(renderResult.error)
    const measuredBounds = await measureMermaidSvgBounds(page, renderResult.svg)
    await page.$eval(
      '#capture',
      (element, svg) => {
        element.innerHTML = svg
      },
      normalizeRenderedMermaidSvg(renderResult.svg, measuredBounds)
    )
    await page.evaluateHandle('document.fonts && document.fonts.ready')

    const box = await page.$eval('#capture', (element) => {
      const rect = element.getBoundingClientRect()
      return {
        width: Math.ceil(rect.width),
        height: Math.ceil(rect.height)
      }
    })

    await page.setViewport({
      width: Math.min(Math.max(box.width + 24, 360), 2400),
      height: Math.min(Math.max(box.height + 24, 220), 3200),
      deviceScaleFactor: 2
    })

    const element = await page.$('#capture')
    if (!element) throw new Error('Rendered Mermaid element was not found.')
    await element.screenshot({ path: outputPath, omitBackground: false })
    await page.close()
  })
}

async function main() {
  const [mode, inputPath, outputPath] = process.argv.slice(2)
  if (!mode || !inputPath || !outputPath) {
    throw new Error(
      'Usage: node scripts/render-book-asset.mjs <svg|mermaid> <input> <output.png>'
    )
  }

  if (mode === 'svg') {
    await renderSvg(path.resolve(inputPath), path.resolve(outputPath))
    return
  }

  if (mode === 'mermaid') {
    await renderMermaid(path.resolve(inputPath), path.resolve(outputPath))
    return
  }

  throw new Error(`Unsupported render mode: ${mode}`)
}

main().catch((error) => {
  console.error(error?.stack || error)
  process.exitCode = 1
})
