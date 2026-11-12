<script setup>
import { computed, ref, watchEffect } from 'vue'

const props = defineProps({
  text: {
    type: [String, Array],
    required: true
  },
  typingSpeed: {
    type: Number,
    default: 46
  },
  deletingSpeed: {
    type: Number,
    default: 22
  },
  pauseDuration: {
    type: Number,
    default: 2200
  },
  initialDelay: {
    type: Number,
    default: 0
  },
  loop: {
    type: Boolean,
    default: true
  },
  cursorCharacter: {
    type: String,
    default: '|'
  }
})

const textArray = computed(() =>
  Array.isArray(props.text) ? props.text : [props.text]
)

const displayedText = ref('')
const currentIndex = ref(0)
const currentCharIndex = ref(0)
const isDeleting = ref(false)

watchEffect((onCleanup) => {
  const items = textArray.value
  if (!items.length) {
    displayedText.value = ''
    return
  }

  const sentence = String(items[currentIndex.value] ?? '')
  const isLastSentence = currentIndex.value === items.length - 1
  let timer

  if (!isDeleting.value && currentCharIndex.value < sentence.length) {
    timer = window.setTimeout(
      () => {
        displayedText.value += sentence[currentCharIndex.value]
        currentCharIndex.value += 1
      },
      currentCharIndex.value === 0 ? props.initialDelay : props.typingSpeed
    )
  } else if (!isDeleting.value && currentCharIndex.value >= sentence.length) {
    if (!props.loop && isLastSentence) return

    timer = window.setTimeout(() => {
      isDeleting.value = true
    }, props.pauseDuration)
  } else if (isDeleting.value && displayedText.value.length > 0) {
    timer = window.setTimeout(() => {
      displayedText.value = displayedText.value.slice(0, -1)
      currentCharIndex.value = Math.max(0, currentCharIndex.value - 1)
    }, props.deletingSpeed)
  } else if (isDeleting.value && displayedText.value.length === 0) {
    timer = window.setTimeout(() => {
      isDeleting.value = false
      currentIndex.value = (currentIndex.value + 1) % items.length
    }, 120)
  }

  onCleanup(() => clearTimeout(timer))
})
</script>

<template>
  <span class="ct-text-type">
    <span class="ct-text-type__content">{{ displayedText }}</span>
    <span class="ct-text-type__cursor">{{ cursorCharacter }}</span>
  </span>
</template>

<style scoped>
.ct-text-type {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.ct-text-type__cursor {
  margin-left: 0.04em;
  opacity: 0.8;
  animation: ct-cursor-blink 0.85s steps(1) infinite;
}

@keyframes ct-cursor-blink {
  0%,
  49% {
    opacity: 0.85;
  }

  50%,
  100% {
    opacity: 0;
  }
}
</style>
