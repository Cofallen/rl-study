export function initGithubStars(theme) {
  if (typeof document === 'undefined') return

  const githubLink = theme.value?.socialLinks?.find(
    (l) => l.icon === 'github' || l.link.includes('github.com')
  )?.link

  if (!githubLink) return

  const match = githubLink.match(/github\.com\/([^/]+\/[^/]+)/)
  if (!match) return

  let repoName = match[1].replace(/\.git$/, '').replace(/\/$/, '')

  fetch(`https://api.github.com/repos/${repoName}`)
    .then((res) => res.json())
    .then((data) => {
      if (data.stargazers_count !== undefined) {
        const stars = data.stargazers_count
        const formatStars = (num) =>
          num > 999 ? (num / 1000).toFixed(1) + 'k' : num
        const starsText = formatStars(stars)

        const injectStars = () => {
          document
            .querySelectorAll('.VPSocialLink[href*="github.com"]')
            .forEach((el) => {
              if (!el.querySelector('.ct-github-stars')) {
                const span = document.createElement('span')
                span.className = 'ct-github-stars'
                span.textContent = starsText

                el.appendChild(span)

                el.style.width = 'auto'
                el.style.padding = '0 8px'
                el.style.textDecoration = 'none'
                el.style.gap = '6px'

                span.style.fontSize = '13px'
                span.style.fontWeight = '500'
                span.style.fontFamily = 'var(--vp-font-family-base)'
              }
            })
        }

        const observer = new MutationObserver(injectStars)
        observer.observe(document.body, { childList: true, subtree: true })
        injectStars()
      }
    })
    .catch(console.error)
}
