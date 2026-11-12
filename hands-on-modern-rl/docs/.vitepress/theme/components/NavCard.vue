<script setup>
import { withBase } from 'vitepress'

defineProps({
  href: {
    type: String,
    required: true
  },
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  },
  icon: {
    type: String,
    default: ''
  }
})
</script>

<template>
  <a :href="withBase(href)" class="ct-nav-card-link">
    <div class="ct-nav-card">
      <div class="ct-nav-card-top">
        <div class="ct-nav-card-header">
          <span v-if="icon" class="ct-nav-card-icon">{{ icon }}</span>
          <span class="ct-nav-card-title">{{ title }}</span>
        </div>
        <span class="ct-nav-card-arrow">↗</span>
      </div>
      <div v-if="description" class="ct-nav-card-desc">{{ description }}</div>
    </div>
  </a>
</template>

<style scoped>
.ct-nav-card-link {
  display: block;
  color: inherit !important;
  text-decoration: none !important;
  border-bottom: 0 !important;
}

.ct-nav-card {
  height: 100%;
  min-height: 126px;
  padding: 20px 22px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: 24px;
  background: linear-gradient(
    180deg,
    rgba(255, 255, 255, 0.95),
    rgba(248, 250, 252, 0.96)
  );
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
  transition:
    transform 0.25s ease,
    box-shadow 0.25s ease,
    border-color 0.25s ease;
}

.ct-nav-card:hover {
  transform: translateY(-4px);
  border-color: rgba(15, 118, 110, 0.18);
  box-shadow: 0 22px 44px rgba(15, 118, 110, 0.12);
}

.ct-nav-card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.ct-nav-card-header {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.ct-nav-card-icon {
  font-size: 22px;
}

.ct-nav-card-title {
  font-size: 17px;
  font-weight: 700;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

.ct-nav-card-arrow {
  margin-top: 2px;
  font-size: 16px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  transition: transform 0.25s ease;
}

.ct-nav-card:hover .ct-nav-card-arrow {
  transform: translate(2px, -2px);
}

.ct-nav-card-desc {
  margin-top: 10px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

.dark .ct-nav-card {
  border-color: rgba(148, 163, 184, 0.16);
  background: linear-gradient(
    180deg,
    rgba(15, 23, 42, 0.9),
    rgba(30, 41, 59, 0.86)
  );
  box-shadow: none;
}

@media (max-width: 768px) {
  .ct-nav-card {
    min-height: 108px;
    padding: 16px 18px;
    border-radius: 18px;
  }

  .ct-nav-card-title {
    font-size: 16px;
  }

  .ct-nav-card-desc {
    font-size: 13px;
  }
}
</style>
