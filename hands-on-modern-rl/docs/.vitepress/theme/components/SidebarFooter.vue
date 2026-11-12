<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'
import { Github, MessageCircle, Moon, Sun } from 'lucide-vue-next'

const { isDark, lang, theme } = useData()

const isEnglish = computed(() => lang.value.startsWith('en'))
const appearanceLabel = computed(() => {
  if (isEnglish.value) {
    return isDark.value ? 'Switch to light mode' : 'Switch to dark mode'
  }
  return isDark.value ? '切换到浅色' : '切换到深色'
})

function toggleAppearance() {
  isDark.value = !isDark.value
}

const githubUrl = computed(() => {
  const repo = theme.value.editLink?.repo
  if (repo) return `https://github.com/${repo}`
  return 'https://github.com/walkinglabs/hands-on-modern-rl'
})

const discordUrl = 'https://discord.gg/XU7DQmpqk'
</script>

<template>
  <div class="ct-sidebar-footer">
    <div class="ct-sidebar-footer-divider" />
    <div class="ct-sidebar-footer-row">
      <div class="ct-sidebar-footer-actions">
        <button
          class="ct-sidebar-footer-btn"
          type="button"
          :title="appearanceLabel"
          :aria-label="appearanceLabel"
          @click="toggleAppearance"
        >
          <Sun v-if="isDark" :size="16" :stroke-width="2" aria-hidden="true" />
          <Moon v-else :size="16" :stroke-width="2" aria-hidden="true" />
        </button>
        <slot name="settings" />
      </div>
      <a
        class="ct-sidebar-footer-link"
        :href="discordUrl"
        target="_blank"
        rel="noopener noreferrer"
        title="Discord"
        aria-label="Discord"
      >
        <MessageCircle :size="16" :stroke-width="2" aria-hidden="true" />
      </a>
      <a
        class="ct-sidebar-footer-link"
        :href="githubUrl"
        target="_blank"
        rel="noopener noreferrer"
        title="GitHub"
        aria-label="GitHub"
      >
        <Github :size="16" :stroke-width="2" aria-hidden="true" />
      </a>
    </div>
  </div>
</template>
