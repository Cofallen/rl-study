import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const rootDir = path.resolve(path.dirname(__filename), '..')
const docsDir = path.join(rootDir, 'docs')
const renderBookAssetScriptPath = path.join(
  rootDir,
  'scripts',
  'render-book-asset.mjs'
)
const configPath = path.join(docsDir, '.vitepress', 'config.mjs')
const packageJsonPath = path.join(rootDir, 'package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const workDir = path.join(rootDir, 'temp', 'latex-book')
const assetDir = path.join(workDir, 'assets')
const distDir = path.join(docsDir, '.vitepress', 'dist')
const texPath = path.join(workDir, 'book.tex')

const pdfLocale = String(process.env.PDF_LOCALE || 'zh').toLowerCase()
const isEnglishPdf = pdfLocale === 'en'
const pdfVersion = process.env.PDF_VERSION || '0.1.0'
const pdfVersionLabel = versionLabel(pdfVersion)
const sourcePageLimit = Math.max(0, readNumberArg('--limit', 18))
const defaultPdfFileName =
  sourcePageLimit > 0
    ? `hands-on-modern-rl-open-textbook-preview-${pdfVersionLabel}.pdf`
    : `hands-on-modern-rl-open-textbook-${pdfVersionLabel}.pdf`
const pdfOutputPath = path.join(
  distDir,
  process.env.LATEX_BOOK_FILE_NAME || defaultPdfFileName
)
const walkingLabsGithubUrl =
  process.env.PDF_WALKINGLABS_GITHUB_URL || 'https://github.com/walkinglabs'
const sanbuGithubUrl =
  process.env.PDF_SANBU_GITHUB_URL || 'https://github.com/sanbuphy'
const pdfAuthors =
  process.env.PDF_AUTHORS ||
  process.env.PDF_AUTHOR ||
  `WalkingLabs; 散步 (${sanbuGithubUrl})`
const pdfTitleLogoWidth = process.env.PDF_LOGO_WIDTH || '118mm'
const pdfPaperWidth = process.env.PDF_PAPER_WIDTH || '210mm'
const pdfPaperHeight = process.env.PDF_PAPER_HEIGHT || '297mm'
const pdfBodyFontSize = process.env.PDF_BODY_FONT_SIZE || '9'
const pdfBodyLineHeight = process.env.PDF_BODY_LINE_HEIGHT || '10.8'
const pdfImageWidth = process.env.PDF_IMAGE_WIDTH || '0.96\\linewidth'
const pdfImageMaxHeight = process.env.PDF_IMAGE_MAX_HEIGHT || '0.42\\textheight'
const pdfDiagramImageWidth =
  process.env.PDF_DIAGRAM_IMAGE_WIDTH || '0.98\\linewidth'
const pdfDiagramImageMaxHeight =
  process.env.PDF_DIAGRAM_IMAGE_MAX_HEIGHT || '0.72\\textheight'
const pdfOptimize = !['0', 'false', 'no', 'off'].includes(
  String(process.env.PDF_OPTIMIZE || '1').toLowerCase()
)
const pdfOptimizeProfile = process.env.PDF_OPTIMIZE_PROFILE || 'default'
const pdfEnglishTitle = process.env.PDF_ENGLISH_TITLE || 'Hands-On Modern RL'
const pdfChineseTitle =
  process.env.PDF_CHINESE_TITLE ||
  (isEnglishPdf
    ? 'Modern Reinforcement Learning in Practice'
    : '现代强化学习实战')
const pdfBookTitle =
  process.env.PDF_BOOK_TITLE ||
  (isEnglishPdf
    ? 'Hands-On Modern RL'
    : `${pdfEnglishTitle}：${pdfChineseTitle}`)
const pdfSubtitle =
  process.env.PDF_SUBTITLE ||
  (isEnglishPdf
    ? 'Modern Reinforcement Learning in Practice — From Code to Theory'
    : '从可运行代码到强化学习原理')
const pdfTagline =
  process.env.PDF_TAGLINE ||
  (isEnglishPdf
    ? 'Open textbook PDF: classic control, LLM post-training, RLVR, Agentic RL, and multimodal agents.'
    : '开放教材 · 书籍版 PDF：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体。')
const pdfCoverMission =
  process.env.PDF_COVER_MISSION ||
  (isEnglishPdf
    ? 'We want more learners to have the courage, possibility, and capability to push toward the frontier of intelligent systems, while keeping the belief that technology should make life better.'
    : '我们希望更多人都有向 AGI 上限发起挑战的勇气、可能性和能力，这会让我们能够解决更多人无法解决的疑难问题，始终相信技术应该让生活更美好。')
const pdfOnlineUrl =
  process.env.PDF_ONLINE_URL ||
  (isEnglishPdf
    ? 'https://walkinglabs.github.io/hands-on-modern-rl/en/'
    : 'https://walkinglabs.github.io/hands-on-modern-rl/')
const pdfLicenseName =
  process.env.PDF_LICENSE_NAME ||
  'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)'
const pdfLicenseUrl =
  process.env.PDF_LICENSE_URL ||
  'https://creativecommons.org/licenses/by-nc-sa/4.0/'
const pdfBuildDate =
  process.env.PDF_BUILD_DATE ||
  new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'long',
    timeZone: 'Asia/Shanghai'
  }).format(new Date())
const keepWorkDir = process.argv.includes('--keep-workdir')
const buildWarnings = []

function versionLabel(value) {
  const clean = String(value || '').trim()
  return /^v/i.test(clean) ? clean : `v${clean || '0.1.0'}`
}

function readArg(name, fallback = null) {
  const equalsArg = process.argv.find((arg) => arg.startsWith(`${name}=`))
  if (equalsArg) return equalsArg.slice(name.length + 1)

  const index = process.argv.indexOf(name)
  if (index !== -1 && process.argv[index + 1]) return process.argv[index + 1]

  return fallback
}

function readNumberArg(name, fallback) {
  const raw = readArg(name, process.env.PDF_BOOK_LIMIT || String(fallback))
  const value = Number(raw)
  return Number.isFinite(value) ? value : fallback
}

function toPosix(value) {
  return value.split(path.sep).join('/')
}

function repositoryUrl() {
  if (process.env.PDF_GITHUB_URL) return process.env.PDF_GITHUB_URL

  const githubRepository = process.env.GITHUB_REPOSITORY
  if (githubRepository && githubRepository.includes('/')) {
    return `https://github.com/${githubRepository.replace(/\.git$/, '')}`
  }

  const repository =
    typeof packageJson.repository === 'string'
      ? packageJson.repository
      : packageJson.repository?.url || ''

  const sshMatch = repository.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
  if (sshMatch) {
    return `https://github.com/${sshMatch[1]}/${sshMatch[2]}`
  }

  const normalized = repository
    .replace(/^git\+/, '')
    .replace(/^https?:\/\/github\.com\//, '')
    .replace(/^git@github\.com:/, '')
    .replace(/\.git$/, '')

  if (normalized.includes('/')) {
    return `https://github.com/${normalized}`
  }

  return 'https://github.com/walkinglabs/hands-on-modern-rl'
}

const pdfGithubUrl = repositoryUrl()

function stripFrontmatter(value) {
  if (!value.startsWith('---\n')) return value

  const end = value.indexOf('\n---\n', 4)
  if (end === -1) return value
  return value.slice(end + 5)
}

function stripScriptSetup(value) {
  return value
    .replace(/^\s*<script\b[\s\S]*?<\/script>\s*/i, '')
    .replace(/<style\b[\s\S]*?<\/style>/gi, '')
}

function stripMarkdown(value) {
  return value
    .replace(/\{#[^}]+\}/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#]/g, '')
    .trim()
}

function stripTitleNumber(value) {
  return stripMarkdown(value)
    .replace(/^第\s*\d+\s*章[：:\s-]*/, '')
    .replace(/^[A-Z]\.\s+/, '')
    .replace(/^\d+(?:\.\d+)*[.、]?\s*/, '')
    .trim()
}

function escapeLatex(value) {
  return String(value)
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([{}%&#_^])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}')
    .replace(/\$/g, '\\$')
}

function escapeLatexUrl(value) {
  return String(value)
    .replace(/\\/g, '/')
    .replace(/([{}%#])/g, '\\$1')
}

function normalizePdfSymbols(value) {
  let output = String(value)
    .replace(/🖼️/g, '图片')
    .replace(/⚠️/g, '注意')
    .replace(/\uFE0F/g, '')
    .replace(/🚧/g, '施工中')
    .replace(/🌟/g, 'Star')
    .replace(/⭐/g, '目标')
    .replace(/🏆/g, '目标')
    .replace(/📁/g, '目录')
    .replace(/📊/g, '图表')
    .replace(/📸/g, '截图')
    .replace(/🏥/g, '医学')
    .replace(/🚀/g, '启动')
    .replace(/✅/g, '完成')
    .replace(/❌/g, '未通过')
    .replace(/👀/g, '查看')
    .replace(/👍/g, '赞成')
    .replace(/👎/g, '反对')
    .replace(/🔴/g, '红色')
    .replace(/🔵/g, '蓝色')
    .replace(/⋮/g, '...')
    .replace(/\bStar\s+Star\b/g, 'Star')

  if (isEnglishPdf) {
    output = output
      .replace(/图片/g, 'Image')
      .replace(/注意/g, 'Note')
      .replace(/目录/g, 'Contents')
      .replace(/启动/g, 'Start')
      .replace(/完成/g, 'Done')
      .replace(/未通过/g, 'Failed')
      .replace(/查看/g, 'View')
      .replace(/赞成/g, 'Approve')
      .replace(/反对/g, 'Reject')
      .replace(/红色/g, 'Red')
      .replace(/蓝色/g, 'Blue')
  }

  return output
}

function warnOnce(message) {
  if (!buildWarnings.includes(message)) buildWarnings.push(message)
}

function runTool(command, args) {
  const result = spawnSync(command, args, {
    cwd: rootDir,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  })

  return result.status === 0
}

const rawLatexTokens = []

function rawLatex(value) {
  const key = `RAWLATEX${rawLatexTokens.length}END`
  rawLatexTokens.push(String(value))
  return key
}

function protectLatexTokens() {
  const tokens = []

  return {
    token(value) {
      const key = `LATEXTOKEN${tokens.length}END`
      tokens.push([key, value])
      return key
    },
    restore(value) {
      let output = value
      for (let pass = 0; pass < 6; pass += 1) {
        const previous = output
        for (const [key, tokenValue] of [...tokens].reverse()) {
          output = output.replaceAll(key, tokenValue)
        }
        if (output === previous) break
      }
      return output
    }
  }
}

function normalizeLatexMath(value) {
  return String(value)
    .replace(/\\\\([A-Za-z])/g, (match, commandStart) => `\\${commandStart}`)
    .replace(/\\\*/g, '*')
}

function renderHref(label, target) {
  const cleanLabel = normalizePdfSymbols(normalizeInlineHtml(label))
    .replace(/\s+/g, ' ')
    .trim()
  const isUrlLike =
    /^https?:\/\//i.test(cleanLabel) ||
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanLabel) ||
    /^mailto:/i.test(target)

  if (isUrlLike) {
    const display = cleanLabel.replace(/^mailto:/i, '')
    return `\\href{${escapeLatexUrl(target)}}{\\nolinkurl{${escapeLatexUrl(
      display
    )}}}`
  }

  return `\\href{${escapeLatexUrl(target)}}{${renderInline(label)}}`
}

function renderInline(value) {
  const protector = protectLatexTokens()
  let source = normalizeInlineHtml(String(value || ''))

  source = source.replace(/RAWLATEX(\d+)END/g, (match, index) => {
    const raw = rawLatexTokens[Number(index)]
    return raw === undefined ? match : protector.token(raw)
  })

  source = source.replace(/`([^`]+)`/g, (match, code) =>
    protector.token(`\\texttt{${escapeLatex(code)}}`)
  )

  source = source.replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g, (match, math) =>
    protector.token(`$${normalizeLatexMath(math)}$`)
  )

  source = source.replace(/!\[([^\]]*)\]\([^)]+\)/g, (match, alt) => alt || '')

  source = source.replace(
    /\[([^\]]+)\]\(((?:https?:\/\/|mailto:)(?:[^()\s]+|\([^)]*\))+)\)/g,
    (match, label, target) => protector.token(renderHref(label, target))
  )

  source = source.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')

  source = source.replace(/\*\*([^*]+)\*\*/g, (match, strong) =>
    protector.token(`\\textbf{${renderInline(strong)}}`)
  )

  source = source.replace(
    /(^|[^\w])_([^_\n]+)_($|[^\w])/g,
    (match, before, em, after) =>
      `${before}${protector.token(`\\emph{${renderInline(em)}}`)}${after}`
  )

  source = source.replace(/\*([^*\n]+)\*/g, (match, em) =>
    protector.token(`\\emph{${renderInline(em)}}`)
  )

  return protector.restore(escapeLatex(source))
}

function renderAuthors() {
  if (process.env.PDF_AUTHORS || process.env.PDF_AUTHOR) {
    return renderInline(pdfAuthors)
  }

  return `\\href{${escapeLatexUrl(
    walkingLabsGithubUrl
  )}}{WalkingLabs}; \\href{${escapeLatexUrl(sanbuGithubUrl)}}{散步}`
}

function normalizeInlineHtml(value) {
  return normalizePdfSymbols(
    value
      .replace(/<br\s*\/?>/gi, ' ')
      .replace(
        /<a\b[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
        '[$2]($1)'
      )
      .replace(/<\/?strong\b[^>]*>/gi, '**')
      .replace(/<\/?b\b[^>]*>/gi, '**')
      .replace(/<\/?em\b[^>]*>/gi, '*')
      .replace(/<\/?i\b[^>]*>/gi, '*')
      .replace(/<code\b[^>]*>([\s\S]*?)<\/code>/gi, '`$1`')
      .replace(/<[^>]+>/g, '')
  )
}

function renderParagraph(lines) {
  const text = lines
    .map((line) => normalizeInlineHtml(line.trim()))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()

  if (!text) return ''
  return `${renderInline(text)}\n`
}

function renderHeading(level, title) {
  const cleanTitle = stripTitleNumber(title.replace(/\s+\{#[^}]+\}\s*$/, ''))
  if (!cleanTitle) return ''

  if (level <= 1) return `\\section{${renderInline(cleanTitle)}}\n`
  if (level === 2) return `\\subsection{${renderInline(cleanTitle)}}\n`
  if (level === 3) return `\\subsubsection{${renderInline(cleanTitle)}}\n`
  return `\\paragraph{${renderInline(cleanTitle)}}\n`
}

function renderBlockquote(lines) {
  const body = lines
    .map((line) => line.replace(/^>\s?/, '').trim())
    .filter(Boolean)
    .map(renderInline)
    .join('\n\n')

  if (!body) return ''
  return `\\begin{quote}\n${body}\n\\end{quote}\n`
}

function renderContainerTitle(kind, rawTitle = '') {
  const title = stripMarkdown(normalizeInlineHtml(rawTitle)).trim()
  const normalizedKind = String(kind || '').toLowerCase()

  if (normalizedKind === 'code-group')
    return isEnglishPdf ? 'Code group' : '代码组'
  if (title && !/^note$/i.test(title)) return title

  const fallbackTitles = isEnglishPdf
    ? {
        info: 'Info',
        note: 'Note',
        tip: 'Tip',
        warning: 'Warning',
        danger: 'Warning',
        details: 'Details'
      }
    : {
        info: '说明',
        note: '说明',
        tip: '提示',
        warning: '注意',
        danger: '警告',
        details: '补充说明'
      }

  return (
    fallbackTitles[normalizedKind] ||
    title ||
    normalizedKind ||
    (isEnglishPdf ? 'Note' : '说明')
  )
}

function renderList(lines, ordered) {
  const env = ordered ? 'enumerate' : 'itemize'
  const items = lines
    .map((line) =>
      line.replace(/^\s*(?:[-*+]|\d+[.)])\s+/, '').replace(/\s+$/, '')
    )
    .filter(Boolean)
    .map((line) => `\\item ${renderInline(line)}`)
    .join('\n')

  if (!items) return ''
  return `\\begin{${env}}\n${items}\n\\end{${env}}\n`
}

function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  const cells = []
  let cell = ''
  let inMath = false
  let inCode = false

  for (let index = 0; index < trimmed.length; index += 1) {
    const char = trimmed[index]
    const previous = trimmed[index - 1]

    if (char === '`' && previous !== '\\') inCode = !inCode
    if (char === '$' && previous !== '\\' && !inCode) inMath = !inMath

    if (char === '|' && !inMath && !inCode) {
      cells.push(cell.trim())
      cell = ''
      continue
    }

    cell += char
  }

  cells.push(cell.trim())
  return cells
}

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line)
}

function renderTable(lines) {
  if (lines.length < 2 || !isTableDivider(lines[1])) return ''

  const rows = [splitTableRow(lines[0]), ...lines.slice(2).map(splitTableRow)]
  const columnCount = Math.max(...rows.map((row) => row.length))
  const columns = `|${Array.from({ length: columnCount }, () => 'X').join('|')}|`
  const body = rows
    .map((row, index) => {
      const cells = Array.from({ length: columnCount }, (_, cellIndex) =>
        renderInline(row[cellIndex] || '')
      )
      const suffix = index === 0 ? ' \\\\ \\hline\\hline' : ' \\\\ \\hline'
      return `${cells.join(' & ')}${suffix}`
    })
    .join('\n')

  return [
    '\\begin{center}',
    '\\scriptsize',
    '\\renewcommand{\\arraystretch}{1.16}',
    `\\begin{tabularx}{\\linewidth}{${columns}}`,
    '\\hline',
    body,
    '\\end{tabularx}',
    '\\end{center}'
  ].join('\n')
}

function renderMermaidBlock(lines) {
  assetCount += 1
  const assetStem = `asset-${String(assetCount).padStart(4, '0')}`
  const diagramPath = path.join(assetDir, `${assetStem}.mmd`)
  const targetPath = path.join(assetDir, `${assetStem}.png`)
  fs.writeFileSync(diagramPath, lines.join('\n'))

  if (renderBookAsset('mermaid', diagramPath, targetPath)) {
    const relative = toPosix(path.relative(workDir, targetPath))
    return [
      '\\begin{figure}[htbp]',
      '\\centering',
      `\\includegraphics[width=${pdfDiagramImageWidth},height=${pdfDiagramImageMaxHeight},keepaspectratio]{${relative}}`,
      '\\end{figure}\n'
    ].join('\n')
  }

  warnOnce(
    isEnglishPdf
      ? 'Unable to render a Mermaid diagram; kept a static note in the PDF.'
      : '无法渲染 Mermaid 图，已在正文中保留静态说明。'
  )
  return [
    `\\begin{BookNote}{${isEnglishPdf ? 'Diagram' : '图示'}}`,
    isEnglishPdf
      ? 'The source page contains a Mermaid diagram. The build environment could not prerender it, so this PDF keeps a textual placeholder.'
      : '这里原文包含一张 Mermaid 流程图；构建环境未能完成预渲染，PDF 中暂以文字说明保留位置。',
    '\\end{BookNote}\n'
  ].join('\n')
}

function renderCodeBlock(language, lines) {
  if (language === 'mermaid') {
    return renderMermaidBlock(lines)
  }

  const code = lines
    .join('\n')
    .replace(/\\end\{Verbatim\}/g, '\\end {Verbatim}')
  return [
    '\\begin{Verbatim}[fontsize=\\scriptsize,frame=single,framesep=1.4mm]',
    code,
    '\\end{Verbatim}\n'
  ].join('\n')
}

function renderImage(sourceFile, alt, target) {
  const resolved = resolveMarkdownAsset(sourceFile, target)
  const label = alt ? renderInline(alt) : '图'

  if (!resolved) {
    return [
      `\\begin{BookNote}{${isEnglishPdf ? 'Image omitted' : '图略'}}`,
      isEnglishPdf
        ? `${label}. External or unsupported image resource: \\url{${escapeLatexUrl(target)}}`
        : `${label}。外部或暂不支持的图片资源：\\url{${escapeLatexUrl(target)}}`,
      '\\end{BookNote}\n'
    ].join('\n')
  }

  const copied = copyAsset(resolved)
  if (!copied) {
    return [
      `\\begin{BookNote}{${isEnglishPdf ? 'Image omitted' : '图略'}}`,
      isEnglishPdf
        ? `${label}. This image format cannot be embedded directly yet: \\texttt{${escapeLatex(path.basename(resolved))}}`
        : `${label}。暂不支持直接嵌入该图片格式：\\texttt{${escapeLatex(path.basename(resolved))}}`,
      '\\end{BookNote}\n'
    ].join('\n')
  }

  return [
    '\\begin{figure}[htbp]',
    '\\centering',
    `\\includegraphics[width=${pdfImageWidth},height=${pdfImageMaxHeight},keepaspectratio]{${copied}}`,
    `\\caption*{\\footnotesize ${label}}`,
    '\\end{figure}\n'
  ].join('\n')
}

function parseMarkdownImageLine(line) {
  const match = line.match(/^!\[([^\]]*)\]\(/)
  if (!match) return null

  const targetStart = match[0].length
  let depth = 0
  let targetEnd = -1

  for (let index = targetStart; index < line.length; index += 1) {
    const char = line[index]
    const previous = line[index - 1]

    if (char === '(' && previous !== '\\') {
      depth += 1
      continue
    }

    if (char === ')' && previous !== '\\') {
      if (depth === 0) {
        targetEnd = index
        break
      }
      depth -= 1
    }
  }

  if (targetEnd === -1) return null

  const rawTarget = line.slice(targetStart, targetEnd).trim()
  const target = rawTarget.replace(/\s+["'][^"']*["']$/, '')

  return {
    alt: match[1],
    target
  }
}

function parseHtmlImageLine(line) {
  const srcMatch = line.match(/<img\b[^>]*\bsrc=(["'])([^"']+)\1[^>]*>/i)
  if (!srcMatch) return null

  const altMatch = line.match(/<img\b[^>]*\balt=(["'])([^"']*)\1[^>]*>/i)
  return {
    alt: altMatch?.[2] || '图',
    target: srcMatch[2]
  }
}

function isNavigationTailLine(line) {
  const text = stripMarkdown(normalizeInlineHtml(line))
    .replace(/\s+/g, ' ')
    .trim()

  if (!text) return false
  if (/^←\s*上一节/.test(text) && /下一节/.test(text)) return true
  if (/^上一节[:：]/.test(text) && /下一节[:：]/.test(text)) return true
  if (/^下一节[:：]/.test(text) && /\[[^\]]+\]\(/.test(line)) return true
  return false
}

function isHorizontalRule(line) {
  return /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)
}

function isDisplayMathBoundary(line) {
  return isHorizontalRule(line) || /^#{1,6}\s+/.test(line)
}

function renderCustomComponent(line) {
  const name =
    line.match(/^<\/?([A-Z][A-Za-z0-9_:-]*)/)?.[1] ||
    (isEnglishPdf ? 'component' : '组件')
  return [
    `\\begin{BookNote}{${isEnglishPdf ? 'Interactive component' : '交互组件'}}`,
    isEnglishPdf
      ? `The web page contains an interactive ${renderInline(name)} component here. The PDF keeps a static note; use the online version or repository source for interactive demos.`
      : `原网页这里包含 ${renderInline(name)} 交互组件。PDF 书籍版保留静态说明；需要运行交互演示时，请回到在线版本或仓库源码。`,
    '\\end{BookNote}\n'
  ].join('\n')
}

function collectFootnotes(markdown) {
  const lines = markdown.split('\n')
  const output = []
  const footnotes = new Map()

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const match = line.match(/^\[\^([^\]]+)\]:\s*(.*)$/)
    if (!match) {
      output.push(line)
      continue
    }

    const noteLines = [match[2]]
    while (index + 1 < lines.length && /^(?: {2,}|\t)/.test(lines[index + 1])) {
      noteLines.push(lines[index + 1].trim())
      index += 1
    }
    footnotes.set(match[1], noteLines.join(' ').trim())
  }

  return {
    markdown: output.join('\n'),
    footnotes
  }
}

function renderMarkdown(markdown, sourceFile) {
  const { markdown: withoutFootnoteDefinitions, footnotes } = collectFootnotes(
    stripScriptSetup(stripFrontmatter(markdown))
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/\r\n/g, '\n')
  )
  const lines = withoutFootnoteDefinitions.split('\n')
  const output = []
  let index = 0
  let skippedFirstHeading = false

  function footnoteLatex(id) {
    const note = footnotes.get(id)
    return note ? `\\footnote{${renderInline(note)}}` : ''
  }

  function withFootnotes(line) {
    return line.replace(/\[\^([^\]]+)\]/g, (match, id) => {
      const note = footnoteLatex(id)
      return note ? rawLatex(note) : ''
    })
  }

  function extractHeadingFootnotes(line) {
    const notes = []
    const title = line
      .replace(/\[\^([^\]]+)\]/g, (match, id) => {
        const note = footnoteLatex(id)
        if (note) notes.push(note)
        return ''
      })
      .replace(/\s+/g, ' ')
      .trim()

    return { title, notes }
  }

  function isBlockStart(line) {
    return (
      /^#{1,6}\s+/.test(line) ||
      isHorizontalRule(line) ||
      /^>\s?/.test(line) ||
      /^\s*(?:[-*+]|\d+[.)])\s+/.test(line) ||
      /^(`{3,}|~{3,})/.test(line) ||
      /^\s*:::\s*/.test(line) ||
      /^!\[[^\]]*\]\([^)]+\)/.test(line) ||
      /^\s*\|.*\|\s*$/.test(line) ||
      /^\s*<\/?div\b/i.test(line) ||
      /^\s*<\/?p\b/i.test(line) ||
      /^\s*<\/?[A-Z][A-Za-z0-9_:-]*/.test(line) ||
      /^\s*\$\$\s*$/.test(line)
    )
  }

  while (index < lines.length) {
    const rawLine = lines[index]
    const line = rawLine.trimEnd()

    if (!line.trim()) {
      index += 1
      continue
    }

    if (isHorizontalRule(line)) {
      index += 1
      continue
    }

    const fence = line.match(/^(`{3,}|~{3,})\s*([A-Za-z0-9_-]*)/)
    if (fence) {
      const marker = fence[1][0]
      const fenceLength = fence[1].length
      const language = (fence[2] || '').toLowerCase()
      const codeLines = []
      index += 1

      while (index < lines.length) {
        const close = lines[index].match(/^(`{3,}|~{3,})/)
        if (close && close[1][0] === marker && close[1].length >= fenceLength) {
          index += 1
          break
        }
        codeLines.push(lines[index])
        index += 1
      }

      output.push(renderCodeBlock(language, codeLines))
      continue
    }

    const oneLineDisplayMath = line.match(/^\s*\$\$(.+)\$\$\s*$/)
    if (oneLineDisplayMath) {
      output.push(`\\[\n${normalizeLatexMath(oneLineDisplayMath[1])}\n\\]\n`)
      index += 1
      continue
    }

    if (/^\s*\$\$\s*$/.test(line)) {
      const mathLines = []
      let cursor = index + 1
      let foundClose = false
      let invalidBlock = false

      while (cursor < lines.length) {
        if (/^\s*\$\$\s*$/.test(lines[cursor])) {
          foundClose = true
          break
        }

        if (isDisplayMathBoundary(lines[cursor])) {
          invalidBlock = true
          break
        }

        mathLines.push(lines[cursor])
        cursor += 1
      }

      if (!foundClose || invalidBlock) {
        warnOnce(
          `跳过疑似未闭合的显示公式围栏：${path.relative(rootDir, sourceFile)}`
        )
        index += 1
        continue
      }

      index = cursor + 1
      const mathBody = mathLines.join('\n').trim()
      if (!mathBody) {
        continue
      }

      output.push(`\\[\n${normalizeLatexMath(mathBody)}\n\\]\n`)
      continue
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) {
      const level = heading[1].length
      if (level === 1 && !skippedFirstHeading) {
        skippedFirstHeading = true
        index += 1
        continue
      }

      const { title, notes } = extractHeadingFootnotes(heading[2])

      // Remove empty "参考文献" or "参考资料" heading because we convert footnotes into LaTeX \footnote{}
      if (
        /^(?:参考文献|参考资料|延伸阅读与参考资料)$/.test(
          stripMarkdown(title).trim()
        )
      ) {
        index += 1
        continue
      }

      output.push(renderHeading(level, title))
      if (notes.length) output.push(`\\mbox{}${notes.join('')}\n`)
      index += 1
      continue
    }

    const image = parseMarkdownImageLine(line)
    if (image) {
      output.push(renderImage(sourceFile, image.alt, image.target))
      index += 1
      continue
    }

    const htmlImage = parseHtmlImageLine(line)
    if (htmlImage) {
      output.push(renderImage(sourceFile, htmlImage.alt, htmlImage.target))
      index += 1
      continue
    }

    if (isNavigationTailLine(line)) {
      index += 1
      continue
    }

    if (/^>\s?/.test(line)) {
      const quoteLines = []
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quoteLines.push(withFootnotes(lines[index]))
        index += 1
      }
      output.push(renderBlockquote(quoteLines))
      continue
    }

    if (/^\s*(?:[-*+])\s+/.test(line)) {
      const listLines = []
      while (index < lines.length && /^\s*(?:[-*+])\s+/.test(lines[index])) {
        listLines.push(withFootnotes(lines[index]))
        index += 1
      }
      output.push(renderList(listLines, false))
      continue
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const listLines = []
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
        listLines.push(withFootnotes(lines[index]))
        index += 1
      }
      output.push(renderList(listLines, true))
      continue
    }

    if (/^\s*\|.*\|\s*$/.test(line) && index + 1 < lines.length) {
      const tableLines = []
      while (index < lines.length && /^\s*\|.*\|\s*$/.test(lines[index])) {
        tableLines.push(withFootnotes(lines[index]))
        index += 1
      }
      const table = renderTable(tableLines)
      if (table) {
        output.push(`${table}\n`)
        continue
      }

      output.push(
        renderParagraph(tableLines.map((tableLine) => tableLine.trim()))
      )
      continue
    }

    if (/^\s*:::\s*/.test(line)) {
      const match = line.match(/^\s*:::\s*([A-Za-z][A-Za-z0-9_-]*)?\s*(.*)$/)
      if (match?.[1]) {
        const title = renderContainerTitle(match[1], match[2])
        output.push(`\\begin{BookNote}{${renderInline(title)}}\n`)
      } else {
        output.push('\\end{BookNote}\n')
      }
      index += 1
      continue
    }

    if (/^\s*<\/?div\b/i.test(line)) {
      index += 1
      continue
    }

    if (/^\s*<\/?p\b/i.test(line)) {
      const paragraph = normalizeInlineHtml(line).trim()
      if (paragraph)
        output.push(
          `\\begin{center}\\footnotesize ${renderInline(paragraph)}\\end{center}\n`
        )
      index += 1
      continue
    }

    if (/^\s*<\/?[A-Z][A-Za-z0-9_:-]*/.test(line)) {
      output.push(renderCustomComponent(line))
      index += 1
      continue
    }

    const paragraphLines = []
    while (index < lines.length) {
      const paragraphLine = lines[index]
      if (!paragraphLine.trim()) break
      if (paragraphLines.length > 0 && isBlockStart(paragraphLine.trimEnd())) {
        break
      }
      paragraphLines.push(withFootnotes(paragraphLine))
      index += 1
    }

    output.push(renderParagraph(paragraphLines))
  }

  return output.join('\n')
}

function splitSuffix(value) {
  const hashIndex = value.indexOf('#')
  const queryIndex = value.indexOf('?')
  const suffixIndexes = [hashIndex, queryIndex].filter((entry) => entry >= 0)
  const suffixIndex = suffixIndexes.length ? Math.min(...suffixIndexes) : -1

  if (suffixIndex === -1) return { clean: value, suffix: '' }
  return {
    clean: value.slice(0, suffixIndex),
    suffix: value.slice(suffixIndex)
  }
}

function markdownPathForLink(link) {
  const { clean } = splitSuffix(link)
  const normalized = clean.replace(/^\//, '')
  const candidates = []

  if (!normalized || normalized.endsWith('/')) {
    candidates.push(path.join(docsDir, normalized, 'index.md'))
  } else {
    candidates.push(path.join(docsDir, `${normalized}.md`))
    candidates.push(path.join(docsDir, normalized, 'index.md'))
  }

  return (
    candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0]
  )
}

function resolveMarkdownAsset(sourceFile, target) {
  if (!target || /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(target)) return null
  if (target.startsWith('data:')) return null

  const { clean } = splitSuffix(target)
  if (!clean) return null

  if (clean.startsWith('/')) {
    const publicCandidate = path.join(docsDir, 'public', clean.slice(1))
    if (fs.existsSync(publicCandidate)) return publicCandidate

    const docsCandidate = path.join(docsDir, clean.slice(1))
    if (fs.existsSync(docsCandidate)) return docsCandidate
    return null
  }

  const relativeCandidate = path.resolve(path.dirname(sourceFile), clean)
  if (fs.existsSync(relativeCandidate)) return relativeCandidate
  return null
}

const copiedAssets = new Map()
let assetCount = 0

function hasUsableFile(filePath) {
  try {
    return fs.statSync(filePath).size > 0
  } catch {
    return false
  }
}

function detectRasterFormat(filePath) {
  let header
  try {
    header = fs.readFileSync(filePath, { length: 16 })
  } catch {
    return null
  }

  if (
    header
      .subarray(0, 8)
      .equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
  ) {
    return 'png'
  }
  if (header.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))) {
    return 'jpeg'
  }
  if (header.subarray(0, 4).toString('ascii') === '%PDF') {
    return 'pdf'
  }
  if (
    header.subarray(0, 4).toString('ascii') === 'RIFF' &&
    header.subarray(8, 12).toString('ascii') === 'WEBP'
  ) {
    return 'webp'
  }
  if (header.subarray(0, 3).toString('ascii') === 'GIF') {
    return 'gif'
  }
  if (header.subarray(4, 8).toString('ascii') === 'ftyp') {
    const brand = header.subarray(8, 12).toString('ascii')
    if (brand === 'avif' || brand === 'avis') return 'avif'
  }

  return null
}

function renderBookAsset(mode, inputPath, outputPath) {
  if (!fs.existsSync(renderBookAssetScriptPath)) return false

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    fs.rmSync(outputPath, { force: true })
    if (
      runTool(process.execPath, [
        renderBookAssetScriptPath,
        mode,
        inputPath,
        outputPath
      ]) &&
      hasUsableFile(outputPath)
    ) {
      return true
    }
  }

  return false
}

function convertAnimatedOrWebAsset(sourcePath, targetPath) {
  return (
    runTool('magick', [`${sourcePath}[0]`, targetPath]) ||
    runTool('convert', [`${sourcePath}[0]`, targetPath]) ||
    runTool('ffmpeg', ['-y', '-i', sourcePath, '-frames:v', '1', targetPath]) ||
    runTool('sips', ['-s', 'format', 'png', sourcePath, '--out', targetPath])
  )
}

function convertSvgAsset(sourcePath, pdfTargetPath, pngTargetPath) {
  // SVG rendering engines like librsvg (rsvg-convert) or Inkscape often fail
  // to render complex Matplotlib SVGs correctly (e.g. black blocks or clipping issues).
  // Chrome/Puppeteer is the most reliable way to render SVGs accurately.
  if (renderBookAsset('svg', sourcePath, pngTargetPath)) {
    return pngTargetPath
  }

  if (
    runTool('rsvg-convert', ['-f', 'pdf', '-o', pdfTargetPath, sourcePath]) ||
    runTool('inkscape', [
      sourcePath,
      '--export-type=pdf',
      `--export-filename=${pdfTargetPath}`
    ])
  ) {
    return pdfTargetPath
  }

  if (
    runTool('rsvg-convert', ['-f', 'png', '-o', pngTargetPath, sourcePath]) ||
    runTool('magick', [sourcePath, pngTargetPath]) ||
    runTool('convert', [sourcePath, pngTargetPath]) ||
    runTool('sips', ['-s', 'format', 'png', sourcePath, '--out', pngTargetPath])
  ) {
    return pngTargetPath
  }

  return null
}

function copyAsset(sourcePath) {
  const ext = path.extname(sourcePath).toLowerCase()

  if (copiedAssets.has(sourcePath)) return copiedAssets.get(sourcePath)

  assetCount += 1
  const assetStem = `asset-${String(assetCount).padStart(4, '0')}`
  let targetPath = null

  const rasterFormat = detectRasterFormat(sourcePath)

  if (['webp', 'gif', 'avif'].includes(rasterFormat)) {
    targetPath = path.join(assetDir, `${assetStem}.png`)
    if (!convertAnimatedOrWebAsset(sourcePath, targetPath)) {
      warnOnce(
        `${isEnglishPdf ? 'Unable to convert image to PNG; keeping a placeholder in the body' : '无法把图片转换为 PNG，已在正文中保留占位'}：${path.relative(
          rootDir,
          sourcePath
        )}`
      )
      return null
    }
  } else if (['.png', '.jpg', '.jpeg', '.pdf'].includes(ext)) {
    targetPath = path.join(assetDir, `${assetStem}${ext}`)
    fs.copyFileSync(sourcePath, targetPath)
  } else if (['.gif', '.webp', '.avif'].includes(ext)) {
    targetPath = path.join(assetDir, `${assetStem}.png`)
    if (!convertAnimatedOrWebAsset(sourcePath, targetPath)) {
      warnOnce(
        `无法把图片转换为 PNG，已在正文中保留占位：${path.relative(
          rootDir,
          sourcePath
        )}`
      )
      return null
    }
  } else if (ext === '.svg') {
    targetPath = convertSvgAsset(
      sourcePath,
      path.join(assetDir, `${assetStem}.pdf`),
      path.join(assetDir, `${assetStem}.png`)
    )

    if (!targetPath) {
      warnOnce(
        `无法把 SVG 转换为 PDF/PNG，已在正文中保留占位：${path.relative(
          rootDir,
          sourcePath
        )}`
      )
      return null
    }
  } else {
    warnOnce(
      `不支持的图片格式，已在正文中保留占位：${path.relative(
        rootDir,
        sourcePath
      )}`
    )
    return null
  }

  if (!hasUsableFile(targetPath)) {
    warnOnce(
      `图片处理结果为空，已在正文中保留占位：${path.relative(
        rootDir,
        sourcePath
      )}`
    )
    return null
  }

  const relative = toPosix(path.relative(workDir, targetPath))
  copiedAssets.set(sourcePath, relative)
  return relative
}

function chapterMetaForItem(item) {
  const text = item.text || ''
  const chapter = text.match(/^(\d+)\.(?!\d)\s*(.+)$/)
  if (chapter) {
    return {
      number: chapter[1],
      title: chapter[2].trim(),
      kind: 'chapter'
    }
  }

  const appendix = text.match(/^([A-Z])\.\s+(.+)$/)
  if (appendix) {
    return {
      number: appendix[1],
      title: appendix[2].trim(),
      kind: 'appendix'
    }
  }

  return null
}

function collectBookStructure(sidebar) {
  const chunks = []
  const seenPages = new Set()

  function addPage(item, chapter = null) {
    if (!item.link || item.link.includes('#') || seenPages.has(item.link))
      return

    seenPages.add(item.link)
    const page = {
      type: 'page',
      title: item.text || item.link,
      link: item.link,
      filePath: markdownPathForLink(item.link)
    }

    if (chapter) chapter.pages.push(page)
    else chunks.push(page)
  }

  function visitItem(item, currentChapter = null) {
    const chapterMeta = chapterMetaForItem(item)

    if (chapterMeta) {
      const chapter = {
        type: 'chapter',
        ...chapterMeta,
        pages: []
      }
      chunks.push(chapter)
      addPage(item, chapter)
      for (const child of item.items || []) visitItem(child, chapter)
      return
    }

    addPage(item, currentChapter)
    for (const child of item.items || []) visitItem(child, currentChapter)
  }

  for (const section of sidebar || []) {
    chunks.push({
      type: 'part',
      title: section.text || (isEnglishPdf ? 'Untitled Part' : '未命名部分')
    })

    if (section.link) visitItem(section)
    for (const item of section.items || []) visitItem(item)
  }

  if (sourcePageLimit === 0) return chunks

  const limited = []
  let remainingPages = sourcePageLimit

  for (const chunk of chunks) {
    if (chunk.type === 'page') {
      if (remainingPages <= 0) continue
      limited.push(chunk)
      remainingPages -= 1
      continue
    }

    if (chunk.type === 'part') {
      if (remainingPages > 0) limited.push(chunk)
      continue
    }

    const pages = chunk.pages.slice(0, Math.max(remainingPages, 0))
    if (!pages.length) continue

    limited.push({ ...chunk, pages })
    remainingPages -= pages.length
  }

  return limited
}

function renderChapterIntro(chapter) {
  const pageList = chapter.pages
    .map((page) => `\\item ${renderInline(stripTitleNumber(page.title))}`)
    .join('\n')

  const label =
    chapter.kind === 'appendix'
      ? isEnglishPdf
        ? `Appendix ${chapter.number}`
        : `附录 ${chapter.number}`
      : isEnglishPdf
        ? `Chapter ${chapter.number}`
        : `第 ${chapter.number} 章`
  return [
    `\\BookChapter{${renderInline(chapter.title)}}{${renderInline(label)}}{`,
    isEnglishPdf ? 'Chapter Guide' : '本章导读',
    '}{',
    pageList,
    '}\n'
  ].join('\n')
}

function renderPage(page, chapter = null) {
  const title = stripTitleNumber(page.title)
  const lines = []

  if (!fs.existsSync(page.filePath)) {
    lines.push(`\\section{${renderInline(title)}}`)
    lines.push(
      `\\begin{BookNote}{${isEnglishPdf ? 'Missing page' : '缺失页面'}}`
    )
    lines.push(
      isEnglishPdf
        ? `The sidebar contains this page, but no matching Markdown file was found locally: \\texttt{${escapeLatex(
            path.relative(rootDir, page.filePath)
          )}}`
        : `Sidebar 中包含该页面，但本地没有找到对应 Markdown 文件：\\texttt{${escapeLatex(
            path.relative(rootDir, page.filePath)
          )}}`
    )
    lines.push('\\end{BookNote}\n')
    return lines.join('\n')
  }

  const markdown = fs.readFileSync(page.filePath, 'utf8')
  const sectionCommand = chapter ? '\\section' : '\\chapter*'

  if (chapter) {
    lines.push(`${sectionCommand}{${renderInline(title)}}`)
  } else {
    lines.push('\\begingroup')
    lines.push('\\setcounter{secnumdepth}{0}')
    lines.push('\\cleardoublepage')
    lines.push(
      `\\phantomsection\\addcontentsline{toc}{chapter}{${renderInline(title)}}`
    )
    lines.push(`${sectionCommand}{${renderInline(title)}}`)
  }

  lines.push(renderMarkdown(markdown, page.filePath))
  if (!chapter) {
    lines.push('\\setcounter{secnumdepth}{2}')
    lines.push('\\endgroup')
  }
  return lines.join('\n')
}

function renderBookContent(chunks) {
  const output = []
  let appendixMode = false

  for (const chunk of chunks) {
    if (chunk.type === 'part') {
      output.push(`\\BookPart{${renderInline(chunk.title)}}\n`)
      continue
    }

    if (chunk.type === 'page') {
      output.push(renderPage(chunk))
      continue
    }

    if (chunk.type === 'chapter') {
      if (chunk.kind === 'appendix' && !appendixMode) {
        output.push('\\appendix\n')
        appendixMode = true
      }

      output.push(renderChapterIntro(chunk))
      for (const page of chunk.pages) {
        output.push(renderPage(page, chunk))
      }
    }
  }

  return output.join('\n')
}

function latexPreamble() {
  return String.raw`
\documentclass[10pt,openany,oneside]{book}
\usepackage[
  paperwidth=${pdfPaperWidth},
  paperheight=${pdfPaperHeight},
  top=16mm,
  bottom=18mm,
  inner=18mm,
  outer=18mm,
  headheight=14pt,
  headsep=5mm,
  footskip=8mm
]{geometry}
\usepackage{fontspec}
\defaultfontfeatures{Ligatures=TeX}
\IfFontExistsTF{Songti SC}{\setmainfont{Songti SC}}{%
  \IfFontExistsTF{PingFang SC}{\setmainfont{PingFang SC}}{%
    \IfFontExistsTF{Noto Serif CJK SC}{\setmainfont{Noto Serif CJK SC}}{%
      \IfFontExistsTF{Arial Unicode MS}{\setmainfont{Arial Unicode MS}}{\setmainfont{Helvetica}}%
    }%
  }%
}
\IfFontExistsTF{PingFang SC}{\setmonofont[Scale=0.82]{PingFang SC}}{%
  \IfFontExistsTF{Noto Sans Mono CJK SC}{\setmonofont[Scale=0.82]{Noto Sans Mono CJK SC}}{%
    \IfFontExistsTF{Noto Sans CJK SC}{\setmonofont[Scale=0.82]{Noto Sans CJK SC}}{%
      \IfFontExistsTF{Songti SC}{\setmonofont[Scale=0.82]{Songti SC}}{\setmonofont[Scale=0.82]{Courier New}}%
    }%
  }%
}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{array}
\usepackage{tabularx}
\usepackage{fancyvrb}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{bookmark}
\hypersetup{
  unicode=true,
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=RoyalBlue,
  linktoc=all,
  pdfpagemode=UseOutlines,
  bookmarksopen=true,
  bookmarksnumbered=true,
  pdftitle={${escapeLatex(pdfBookTitle)}},
  pdfauthor={WalkingLabs; sanbuphy}
}
\definecolor{BookInk}{HTML}{222222}
\definecolor{BookMuted}{HTML}{666A73}
\definecolor{BookRule}{HTML}{D6DAE1}
\definecolor{BookSoft}{HTML}{F5F7FA}
\definecolor{RoyalBlue}{HTML}{2457A6}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.14em}
\setlength{\emergencystretch}{2em}
\linespread{1.02}
\AtBeginDocument{\fontsize{${pdfBodyFontSize}}{${pdfBodyLineHeight}}\selectfont}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\tolerance=1600
\hbadness=3000
\raggedbottom
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.95}
\renewcommand{\contentsname}{${isEnglishPdf ? 'Contents' : '目录'}}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\arraystretch}{1.18}
\captionsetup{font=footnotesize,labelformat=empty}
\makeatletter
\def\@makechapterhead#1{%
  \vspace*{20pt}%
  {\parindent \z@ \raggedright \normalfont
    {\Large\color{BookMuted}\bfseries ${isEnglishPdf ? 'Chapter' : '第'} \thechapter ${isEnglishPdf ? '' : '章'}\par}%
    \vspace{10pt}%
    {\Huge\bfseries #1\par}%
    \vspace{16pt}%
  }%
}
\def\@makeschapterhead#1{%
  \vspace*{18pt}%
  {\parindent \z@ \raggedright \normalfont
    {\Huge\bfseries #1\par}%
    \vspace{14pt}%
  }%
}
\makeatother
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize \href{${escapeLatexUrl(pdfGithubUrl)}}{Hands-On Modern RL}}
\fancyhead[R]{\footnotesize \href{https://github.com/walkinglabs}{作者：WalkingLabs}；\href{${escapeLatexUrl(sanbuGithubUrl)}}{散步}}
\fancyfoot[L]{\scriptsize GitHub: \href{${escapeLatexUrl(pdfGithubUrl)}}{walkinglabs/hands-on-modern-rl} · \href{${escapeLatexUrl(sanbuGithubUrl)}}{sanbuphy}}
\fancyfoot[R]{\scriptsize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \fancyfoot[L]{\scriptsize GitHub: \href{${escapeLatexUrl(pdfGithubUrl)}}{walkinglabs/hands-on-modern-rl} · \href{${escapeLatexUrl(sanbuGithubUrl)}}{sanbuphy}}%
  \fancyfoot[R]{\scriptsize\thepage}%
  \renewcommand{\headrulewidth}{0pt}%
}
\newenvironment{BookNote}[1]{%
  \par\medskip
  \begingroup
  \small
  \noindent\textcolor{RoyalBlue}{\rule{2.4pt}{1.2em}}\hspace{5pt}\textbf{#1}\par
  \vspace{2pt}
  \leftskip=1.1em
}{%
  \par\endgroup\medskip
}
\newcommand{\BookPart}[1]{%
  \cleardoublepage
  \phantomsection
  \addcontentsline{toc}{part}{#1}
  {\thispagestyle{plain}\vspace*{0.32\textheight}
  \begin{center}
  {\Large\color{BookMuted}Part\par}
  \vspace{10pt}
  {\huge\bfseries #1\par}
  \end{center}
  \clearpage}
}
\newcommand{\BookChapter}[4]{%
  \cleardoublepage
  \chapter{#1}
  \thispagestyle{plain}
  \vspace{8mm}
  {\Large\bfseries #2\par}
  \vspace{5mm}
  {\large\bfseries #3\par}
  \vspace{2mm}
  ${isEnglishPdf ? 'This chapter contains the following sections. Skim the structure first, then move into the prose and code.' : '本章包含以下小节，建议先扫一遍结构，再进入正文和代码。'}
  \begin{itemize}
  #4
  \end{itemize}
  \clearpage
}
`
}

function renderTitlePage(logoAsset) {
  const openCourseLabel = isEnglishPdf
    ? 'Open Course · Book PDF'
    : '开放课程 · 书籍版 PDF'
  const authorLabel = isEnglishPdf ? 'Authors' : '作者'
  const repoLabel = isEnglishPdf ? 'Repository' : '仓库'
  const versionLabelText = isEnglishPdf ? 'Version' : '版本'
  const coverNote = isEnglishPdf
    ? 'This book is compiled by WalkingLabs Open Course Series as a versioned, citable PDF for offline reading and classroom use.'
    : '本书由 WalkingLabs 开放课程整理为可离线阅读、可引用、可持续修订的书籍版 PDF。'

  return [
    '\\begin{titlepage}',
    '\\thispagestyle{empty}',
    '\\vspace*{6mm}',
    '\\begin{center}',
    logoAsset
      ? `\\includegraphics[width=${pdfTitleLogoWidth}]{${logoAsset}}\\\\[13mm]`
      : '',
    '{\\large WalkingLabs Open Course Series}\\\\[5mm]',
    '\\rule{0.54\\textwidth}{0.4pt}\\\\[8mm]',
    `{\\Huge\\bfseries ${renderInline(pdfEnglishTitle)}}\\\\[5mm]`,
    `{\\LARGE ${renderInline(pdfChineseTitle)}}\\\\[3mm]`,
    `{\\large ${renderInline(pdfSubtitle)}}\\\\[8mm]`,
    `{\\large ${renderInline(openCourseLabel)}}\\\\[4mm]`,
    `{\\large Version ${renderInline(pdfVersion)} · ${renderInline(
      pdfBuildDate
    )}}\\\\[12mm]`,
    `\\parbox{0.78\\textwidth}{\\centering\\large ${renderInline(
      pdfTagline
    )}}\\\\[18mm]`,
    '\\end{center}',
    '\\vfill',
    '\\noindent\\rule{\\textwidth}{0.4pt}\\\\[4mm]',
    `\\noindent\\textbf{${renderInline(authorLabel)}} \\quad ${renderAuthors()}\\\\[3mm]`,
    `\\noindent\\textbf{${renderInline(repoLabel)}} \\quad \\href{${escapeLatexUrl(
      pdfGithubUrl
    )}}{${renderInline(pdfGithubUrl)}}\\\\[3mm]`,
    `\\noindent\\textbf{${renderInline(versionLabelText)}} \\quad ${renderInline(pdfVersion)}\\\\[8mm]`,
    `\\noindent ${renderInline(coverNote)}\\\\[2mm]`,
    `\\noindent{\\small\\color{BookMuted} ${renderInline(pdfCoverMission)}}`,
    '\\end{titlepage}'
  ].join('\n')
}

function renderFrontMatter(pageCount) {
  if (isEnglishPdf) return renderEnglishFrontMatter(pageCount)

  return [
    '\\frontmatter',
    '\\chapter*{版权与版本}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{版权与版本}',
    '\\section*{出版信息}',
    `\\noindent\\textbf{书名} \\quad ${renderInline(
      `${pdfBookTitle}：${pdfSubtitle}`
    )}\\\\[2mm]`,
    `\\noindent\\textbf{版本} \\quad ${renderInline(pdfVersion)}\\\\[2mm]`,
    `\\noindent\\textbf{构建日期} \\quad ${renderInline(
      pdfBuildDate
    )}\\\\[2mm]`,
    `\\noindent\\textbf{作者} \\quad ${renderAuthors()}\\\\[2mm]`,
    `\\noindent\\textbf{项目仓库} \\quad \\href{${escapeLatexUrl(
      pdfGithubUrl
    )}}{${renderInline(pdfGithubUrl)}}\\\\[2mm]`,
    `\\noindent\\textbf{在线版本} \\quad \\href{${escapeLatexUrl(
      pdfOnlineUrl
    )}}{${renderInline(pdfOnlineUrl)}}`,
    '',
    '\\section*{版权与许可}',
    'Copyright \\copyright{} 2026 WalkingLabs and contributors.',
    '',
    `除非页面、图片或代码片段另有说明，本书文字、课程说明与原创图文内容采用 \\href{${escapeLatexUrl(
      pdfLicenseUrl
    )}}{${renderInline(pdfLicenseName)}} 发布。你可以在非商业目的下复制、分发和改编本材料，但必须保留适当署名、标明修改，并以相同协议分发衍生作品。`,
    '',
    '本仓库中的第三方图片、论文截图、外部链接、引用文本、商标名称和部分代码示例可能受其原始作者或项目的许可约束；使用这些材料时，请同时遵守对应来源的许可与引用要求。',
    '',
    '\\section*{推荐引用}',
    '如果你在课程、学习笔记、研究讨论或衍生非商业教育材料中使用本书，建议引用本仓库，并保留版本号。推荐格式如下：',
    '',
    '\\noindent\\textbf{APA-style}',
    `\\begin{quote}\\small WalkingLabs. (2026). \\emph{${renderInline(
      `${pdfBookTitle}: ${pdfSubtitle}`
    )}} (Version ${renderInline(
      pdfVersion
    )}) [Open courseware]. \\href{${escapeLatexUrl(
      pdfGithubUrl
    )}}{\\nolinkurl{${escapeLatexUrl(pdfGithubUrl)}}}.\\end{quote}`,
    '',
    '\\noindent\\textbf{BibTeX}',
    '\\begin{Verbatim}[fontsize=\\scriptsize,frame=single,framesep=1.2mm]',
    '@misc{hands_on_modern_rl,',
    `  title        = {${pdfBookTitle}: ${pdfSubtitle}},`,
    '  author       = {WalkingLabs},',
    '  year         = {2026},',
    `  howpublished = {\\url{${escapeLatexUrl(pdfGithubUrl)}}},`,
    `  note         = {Open courseware repository, Version ${pdfVersion}},`,
    '}',
    '\\end{Verbatim}',
    '',
    '\\section*{勘误与更新}',
    '本 PDF 按 README 与课程目录结构生成，用于离线阅读、课堂分发和版本化引用。在线网页与代码仓库会持续更新；若 PDF、网页和仓库源码存在差异，请以 GitHub 仓库中的最新内容为准。欢迎通过 GitHub Issues 提交勘误、补充材料和改进建议。',
    '',
    '\\section*{免责声明}',
    '本书按“原样”提供，不附带任何明示或暗示担保。强化学习实验可能受依赖版本、硬件、随机种子和外部模型服务影响；运行代码、复现实验或基于本书内容构建系统时，请自行验证结果并承担相应风险。',
    '',
    '\\chapter*{前言}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{前言}',
    `《${pdfBookTitle}》是一门面向现代强化学习实践的开放课程。与传统的“先讲公式，再给黑盒 API”不同，本课程采用“实践优先”的路径：从一行行可运行的代码和直观的训练现象出发，让学习者先看到智能体如何在环境中试错并从奖励中改进行为，再回头深入剖析其背后的状态、价值函数、策略梯度、奖励建模与信用分配等核心数学结构。`,
    '',
    '课程内容跨越经典控制理论，直接连接到当前最前沿的 AI 进展，包括大语言模型（LLM）后训练、偏好对齐（DPO/GRPO）、可验证奖励（RLVR）、多轮工具调用的 Agentic RL 以及视觉语言模型（VLM）强化学习等核心主题。',
    '',
    '我们希望为你铺设一条坚实的阶梯——从解出 CartPole 的第一步，一直通往构建大模型后训练与智能体系统的前沿实践。',
    '',
    '希望本开源教程能够让更多人拥有向智能上限发起攀登的勇气，解决更多通往 AGI 道路上的问题。这也是本书整理为 PDF 的原因之一：让课程不仅能在网页上快速迭代，也能以一本开放教材的形态被保存、分享、批注和带进课堂。',
    '',
    '当前教程仍在快速迭代中。建议优先阅读已经相对稳定的章节；标注为施工或未完成状态的章节可能仍有错误，也欢迎读者通过 GitHub Issues 和 Pull Request 参与修正与完善。',
    '',
    '由于资源稀缺问题，项目正在寻求显卡支持。如果你有可用于课程实验、训练复现或教学验证的显卡使用方式，并愿意支持本开源教程，欢迎联系 \\href{mailto:physicoada@gmail.com}{physicoada@gmail.com}。本书特别适合以下读者：',
    '',
    '\\begin{itemize}',
    '\\item 从监督学习转向强化学习的机器学习工程师；',
    '\\item 准备阅读现代强化学习、LLM 对齐和 Agentic RL 论文的研究人员与学生；',
    '\\item 希望理解 RLHF、DPO、GRPO、RLVR 与后训练系统的大模型从业者；',
    '\\item 喜欢先看代码、实验和可视化，再进入公式推导的自主学习者。',
    '\\end{itemize}',
    '',
    '如果你第一次接触强化学习，可以从 CartPole 和 PPO 的实战章节开始，把公式当作实验现象背后的解释工具；如果你已经熟悉经典 RL，可以直接进入 DPO、GRPO、RLVR 与 Agentic RL 等现代章节，再回到附录补齐数学细节。',
    '',
    '\\chapter*{本版说明}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{本版说明}',
    '\\subsection*{学习目标}',
    '完成本课程后，学习者应能够：',
    '\\begin{itemize}',
    '\\item 实现并解释核心的强化学习循环：环境交互、轨迹收集、奖励反馈、策略更新和评估；',
    '\\item 将 MDP、价值函数、贝尔曼方程、TD 学习、策略梯度和优势估计与具体的训练行为联系起来；',
    '\\item 阅读并修改 DQN、REINFORCE、Actor-Critic、PPO、DPO、GRPO 及相关实现；',
    '\\item 推理大模型（LLM）的后训练流水线，包括 SFT、奖励建模、PPO 风格的 RLHF、DPO 系列方法和可验证奖励（RLVR）训练；',
    '\\item 理解多轮交互与信用分配，构建工具调用、轨迹合成与 Agentic RL 智能体系统；',
    '\\item 将强化学习延伸到 VLM（视觉语言模型）、具身智能与多智能体自我博弈等前沿领域；',
    '\\item 诊断常见的强化学习失败模式，为新的 RL 问题设计合理的算法、工程评测与调试方案。',
    '\\end{itemize}',
    '',
    `本版本汇编 ${pageCount} 个课程页面，包含封面、前言、目录、PDF 书签、章节开篇页、页眉页脚、右下角页码和可点击的仓库信息。`,
    '',
    '为适配书籍阅读，构建脚本会在导出时处理网页内容中的动态图、SVG、Mermaid 图、脚注、表格、局部参考文献和章节导航。网页中的交互组件会在 PDF 中转写为静态说明；需要运行代码或查看交互演示时，请回到在线版本与仓库源码。',
    '',
    '\\cleardoublepage',
    '\\phantomsection',
    '\\pdfbookmark[0]{目录}{toc}',
    '\\tableofcontents',
    '\\mainmatter'
  ].join('\n')
}

function renderEnglishFrontMatter(pageCount) {
  return [
    '\\frontmatter',
    '\\chapter*{Copyright and Version}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{Copyright and Version}',
    '\\section*{Publication Information}',
    `\\noindent\\textbf{Title} \\quad ${renderInline(
      `${pdfBookTitle}: ${pdfSubtitle}`
    )}\\\\[2mm]`,
    `\\noindent\\textbf{Version} \\quad ${renderInline(pdfVersion)}\\\\[2mm]`,
    `\\noindent\\textbf{Build Date} \\quad ${renderInline(
      pdfBuildDate
    )}\\\\[2mm]`,
    `\\noindent\\textbf{Authors} \\quad ${renderAuthors()}\\\\[2mm]`,
    `\\noindent\\textbf{Repository} \\quad \\href{${escapeLatexUrl(
      pdfGithubUrl
    )}}{${renderInline(pdfGithubUrl)}}\\\\[2mm]`,
    `\\noindent\\textbf{Online Version} \\quad \\href{${escapeLatexUrl(
      pdfOnlineUrl
    )}}{${renderInline(pdfOnlineUrl)}}`,
    '',
    '\\section*{License}',
    'Copyright \\copyright{} 2026 WalkingLabs and contributors.',
    '',
    `Unless otherwise noted on a page, image, or code snippet, the original course text and figures are released under \\href{${escapeLatexUrl(
      pdfLicenseUrl
    )}}{${renderInline(pdfLicenseName)}}. You may copy, distribute, and adapt this material for non-commercial purposes, provided that attribution is preserved, changes are indicated, and derivative works are shared under the same license.`,
    '',
    'Third-party images, paper screenshots, external links, quoted text, trademarks, and some code examples may be governed by their original licenses. Please follow the corresponding source license and citation requirements when using them.',
    '',
    '\\section*{Suggested Citation}',
    'If you use this book in courses, notes, research discussions, or derivative non-commercial educational materials, please cite the repository and keep the version number.',
    '',
    '\\noindent\\textbf{APA-style}',
    `\\begin{quote}\\small WalkingLabs. (2026). \\emph{${renderInline(
      `${pdfBookTitle}: ${pdfSubtitle}`
    )}} (Version ${renderInline(
      pdfVersion
    )}) [Open courseware]. \\href{${escapeLatexUrl(
      pdfGithubUrl
    )}}{\\nolinkurl{${escapeLatexUrl(pdfGithubUrl)}}}.\\end{quote}`,
    '',
    '\\noindent\\textbf{BibTeX}',
    '\\begin{Verbatim}[fontsize=\\scriptsize,frame=single,framesep=1.2mm]',
    '@misc{hands_on_modern_rl,',
    `  title        = {${pdfBookTitle}: ${pdfSubtitle}},`,
    '  author       = {WalkingLabs},',
    '  year         = {2026},',
    `  howpublished = {\\url{${escapeLatexUrl(pdfGithubUrl)}}},`,
    `  note         = {Open courseware repository, Version ${pdfVersion}},`,
    '}',
    '\\end{Verbatim}',
    '',
    '\\section*{Errata and Updates}',
    'This PDF is generated from the README and course sidebar structure for offline reading, classroom distribution, and versioned citation. The online site and repository will continue to evolve. If the PDF, website, and repository source differ, treat the latest GitHub repository content as authoritative. Please use GitHub Issues for errata, additions, and improvement suggestions.',
    '',
    '\\section*{Disclaimer}',
    'This book is provided as-is, without warranties of any kind. Reinforcement learning experiments can depend on package versions, hardware, random seeds, and external model services. Validate results before relying on them in your own systems.',
    '',
    '\\chapter*{Preface}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{Preface}',
    `${pdfBookTitle} is an open course for modern reinforcement learning practice. Instead of starting with formulas and hiding the system behind black-box APIs, this course follows a practice-first path: begin with runnable code, training curves, and failure cases, then return to the mathematical structures underneath them.`,
    '',
    "The course moves from classic control to today's frontier: LLM post-training, preference alignment, DPO and GRPO, RL with verifiable rewards, multi-turn tool-use agents, Agentic RL, VLM reinforcement learning, embodied intelligence, and multi-agent self-play.",
    '',
    'The goal is to build a steady ladder: from solving CartPole to understanding and building post-training and agent systems.',
    '',
    'This open textbook is still evolving. Stable chapters are ready for focused reading; chapters marked as under construction may still contain mistakes. Contributions through GitHub Issues and Pull Requests are welcome.',
    '',
    'This book is especially suitable for:',
    '',
    '\\begin{itemize}',
    '\\item machine learning engineers moving from supervised learning into reinforcement learning;',
    '\\item researchers and students preparing to read modern RL, LLM alignment, and Agentic RL papers;',
    '\\item LLM practitioners who want to understand RLHF, DPO, GRPO, RLVR, and post-training systems;',
    '\\item self-learners who prefer to see code, experiments, and visualizations before formulas.',
    '\\end{itemize}',
    '',
    'If this is your first encounter with reinforcement learning, start from CartPole and PPO, and treat formulas as explanations for experimental behavior. If you already know classic RL, you can jump into DPO, GRPO, RLVR, and Agentic RL, then return to the appendices for mathematical details.',
    '',
    '\\chapter*{Edition Notes}',
    '\\phantomsection\\addcontentsline{toc}{chapter}{Edition Notes}',
    '\\subsection*{Learning Goals}',
    'After completing this course, learners should be able to:',
    '\\begin{itemize}',
    '\\item implement and explain the core RL loop: environment interaction, trajectory collection, reward feedback, policy update, and evaluation;',
    '\\item connect MDPs, value functions, Bellman equations, TD learning, policy gradients, and advantage estimation to concrete training behavior;',
    '\\item read and modify DQN, REINFORCE, Actor-Critic, PPO, DPO, GRPO, and related implementations;',
    '\\item reason about LLM post-training pipelines, including SFT, reward modeling, PPO-style RLHF, DPO-family methods, and RLVR;',
    '\\item understand multi-turn interaction and credit assignment, and build tool-use, trajectory-synthesis, and Agentic RL systems;',
    '\\item extend reinforcement learning ideas to VLMs, embodied intelligence, and multi-agent self-play;',
    '\\item diagnose common RL failure modes and design reasonable algorithms, engineering evaluations, and debugging workflows for new RL problems.',
    '\\end{itemize}',
    '',
    `This edition compiles ${pageCount} course pages with a cover, preface, contents, PDF bookmarks, chapter opening pages, headers, footers, page numbers, and clickable repository metadata.`,
    '',
    'For book-style reading, the build script processes animated images, SVGs, Mermaid diagrams, footnotes, tables, local references, and navigation links. Interactive web components are converted to static notes in the PDF; use the online version or repository source to run interactive demos.',
    '',
    '\\cleardoublepage',
    '\\phantomsection',
    '\\pdfbookmark[0]{Contents}{toc}',
    '\\tableofcontents',
    '\\mainmatter'
  ].join('\n')
}

async function loadSidebar() {
  const configModule = await import(pathToFileURL(configPath).href)
  const config = configModule.default
  if (isEnglishPdf) {
    return (
      config.locales?.en?.themeConfig?.sidebar?.['/en/'] ||
      config.locales?.en?.themeConfig?.sidebar?.['/'] ||
      []
    )
  }

  return (
    config.locales?.zh?.themeConfig?.sidebar?.['/'] ||
    config.themeConfig?.sidebar?.['/'] ||
    []
  )
}

function prepareWorkDir() {
  if (!keepWorkDir) fs.rmSync(workDir, { recursive: true, force: true })
  fs.mkdirSync(assetDir, { recursive: true })
  fs.mkdirSync(distDir, { recursive: true })
}

function compileLatex() {
  for (let run = 1; run <= 2; run += 1) {
    const result = spawnSync(
      'xelatex',
      ['-interaction=nonstopmode', '-halt-on-error', 'book.tex'],
      {
        cwd: workDir,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    if (result.status !== 0) {
      const stdoutLogPath = path.join(workDir, `xelatex-run-${run}.stdout.log`)
      const stderrLogPath = path.join(workDir, `xelatex-run-${run}.stderr.log`)
      fs.writeFileSync(stdoutLogPath, result.stdout || '')
      fs.writeFileSync(stderrLogPath, result.stderr || '')

      const tailLines = String(result.stdout || '')
        .split(/\r?\n/)
        .slice(-120)
        .join('\n')
      if (tailLines.trim()) {
        console.error(`\n--- xelatex stdout tail (run ${run}) ---`)
        console.error(tailLines)
        console.error('--- end xelatex stdout tail ---\n')
      }
      if (String(result.stderr || '').trim()) {
        console.error(`\n--- xelatex stderr (run ${run}) ---`)
        console.error(result.stderr)
        console.error('--- end xelatex stderr ---\n')
      }

      throw new Error(`xelatex failed on run ${run}. See ${stdoutLogPath}`)
    }
  }

  fs.copyFileSync(path.join(workDir, 'book.pdf'), pdfOutputPath)
}

function normalizeGhostscriptProfile(value) {
  const clean = String(value || 'default')
    .trim()
    .replace(/^\//, '')
    .toLowerCase()
  const allowed = new Set(['screen', 'ebook', 'printer', 'prepress', 'default'])
  return allowed.has(clean) ? `/${clean}` : '/default'
}

function optimizePdfOutput() {
  if (!pdfOptimize) return

  const optimizedPath = pdfOutputPath.replace(/\.pdf$/i, '.optimized.pdf')
  fs.rmSync(optimizedPath, { force: true })

  const ok = runTool('gs', [
    '-sDEVICE=pdfwrite',
    '-dCompatibilityLevel=1.7',
    `-dPDFSETTINGS=${normalizeGhostscriptProfile(pdfOptimizeProfile)}`,
    '-dColorConversionStrategy=/LeaveColorUnchanged',
    '-dDownsampleColorImages=false',
    '-dDownsampleGrayImages=false',
    '-dDownsampleMonoImages=false',
    '-dDetectDuplicateImages=true',
    '-dCompressFonts=true',
    '-dSubsetFonts=true',
    '-dNOPAUSE',
    '-dBATCH',
    '-dQUIET',
    `-sOutputFile=${optimizedPath}`,
    pdfOutputPath
  ])

  if (!ok || !hasUsableFile(optimizedPath)) {
    fs.rmSync(optimizedPath, { force: true })
    warnOnce('未能用 Ghostscript 优化 PDF，已保留未压缩版本。')
    return
  }

  fs.renameSync(optimizedPath, pdfOutputPath)
}

async function main() {
  prepareWorkDir()

  const logoSource = path.join(docsDir, 'public', 'readme', 'readmelogo.png')
  const logoAsset = fs.existsSync(logoSource) ? copyAsset(logoSource) : null
  const sidebar = await loadSidebar()
  const chunks = collectBookStructure(sidebar)
  const pageCount = chunks.reduce((count, chunk) => {
    if (chunk.type === 'page') return count + 1
    if (chunk.type === 'chapter') return count + chunk.pages.length
    return count
  }, 0)

  const tex = [
    latexPreamble(),
    '\\begin{document}',
    renderTitlePage(logoAsset),
    renderFrontMatter(pageCount),
    renderBookContent(chunks),
    '\\end{document}'
  ].join('\n\n')

  fs.writeFileSync(texPath, tex)
  compileLatex()
  optimizePdfOutput()

  console.log(`LaTeX book PDF written to ${pdfOutputPath}`)
  console.log(`LaTeX source written to ${texPath}`)
  if (buildWarnings.length) {
    console.warn('LaTeX book build warnings:')
    for (const warning of buildWarnings) console.warn(`- ${warning}`)
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
