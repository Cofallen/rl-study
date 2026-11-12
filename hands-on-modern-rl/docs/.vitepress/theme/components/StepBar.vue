<script setup>
defineProps({
  active: {
    type: Number,
    default: 0
  },
  items: {
    type: Array,
    default: () => []
  }
})
</script>

<template>
  <div class="ct-stepbar">
    <div
      v-for="(item, index) in items"
      :key="item.title || index"
      class="ct-stepbar-item"
      :class="{
        'is-active': index === active,
        'is-complete': index < active
      }"
    >
      <div class="ct-stepbar-node">
        <span>{{ index + 1 }}</span>
      </div>
      <div class="ct-stepbar-copy">
        <div class="ct-stepbar-title">{{ item.title }}</div>
        <div v-if="item.description" class="ct-stepbar-desc">
          {{ item.description }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ct-stepbar {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin: 20px 0 28px;
}

.ct-stepbar-item {
  position: relative;
  display: flex;
  gap: 12px;
  padding: 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  background: linear-gradient(
    180deg,
    rgba(255, 255, 255, 0.94),
    rgba(248, 250, 252, 0.94)
  );
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
}

.ct-stepbar-item.is-active {
  border-color: rgba(15, 118, 110, 0.28);
  box-shadow: 0 18px 40px rgba(15, 118, 110, 0.14);
}

.ct-stepbar-item.is-complete {
  border-color: rgba(245, 158, 11, 0.24);
}

.ct-stepbar-node {
  flex: 0 0 auto;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 800;
  color: #fff;
  background: linear-gradient(135deg, var(--vp-c-brand-1), #0f172a);
  box-shadow: 0 10px 20px rgba(15, 118, 110, 0.22);
}

.ct-stepbar-item.is-complete .ct-stepbar-node {
  background: linear-gradient(135deg, #f59e0b, var(--vp-c-brand-1));
}

.ct-stepbar-copy {
  min-width: 0;
}

.ct-stepbar-title {
  font-size: 15px;
  font-weight: 700;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

.ct-stepbar-desc {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.55;
  color: var(--vp-c-text-2);
}

.dark .ct-stepbar-item {
  border-color: rgba(148, 163, 184, 0.18);
  background: linear-gradient(
    180deg,
    rgba(15, 23, 42, 0.88),
    rgba(30, 41, 59, 0.82)
  );
  box-shadow: none;
}

@media (max-width: 768px) {
  .ct-stepbar {
    grid-template-columns: 1fr;
    gap: 10px;
    margin: 16px 0 22px;
  }
}
</style>
