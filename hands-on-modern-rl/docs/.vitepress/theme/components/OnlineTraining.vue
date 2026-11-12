<script setup>
import { computed } from 'vue'
import { useRoute } from 'vitepress'

const props = defineProps({
  studios: {
    type: String,
    required: true
  },
  lang: {
    type: String,
    default: ''
  },
  compact: {
    type: Boolean,
    default: false
  }
})

const route = useRoute()

const catalog = {
  cartpole: {
    badge: '01',
    title: 'CartPole',
    hardware: 'CPU',
    duration: { zh: '30–60 秒', en: '30–60 sec' },
    description: {
      zh: '用 PPO 从零训练倒立摆，实时查看奖励曲线、训练日志和策略回放。',
      en: 'Train CartPole with PPO and inspect the reward curve, live console, and policy replay.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole'
  },
  gymnasium: {
    badge: 'LAB',
    title: 'Gymnasium Playground',
    hardware: 'CPU',
    duration: { zh: '数秒到数分钟', en: 'seconds to minutes' },
    description: {
      zh: '从教学文本、经典控制到游戏环境，切换任务并调整训练参数。',
      en: 'Switch among teaching, classic-control, and game environments, then adjust the training recipe.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium'
  },
  vizdoom: {
    badge: '02',
    title: 'ViZDoom',
    hardware: 'CPU',
    duration: { zh: '1–4 分钟', en: '1–4 min' },
    description: {
      zh: '让 DQN 从第一人称画面学习瞄准、移动、生存和战斗。',
      en: 'Train DQN from first-person pixels on aiming, navigation, survival, and combat tasks.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment02-vizdoom'
  },
  atari: {
    badge: '03',
    title: 'Atari / ALE',
    hardware: 'CPU',
    duration: { zh: '1–5 分钟', en: '1–5 min' },
    description: {
      zh: '在 Pong 等像素游戏上运行短预算 DQN，观察策略怎样开始学习。',
      en: 'Run short-budget DQN experiments on Pong and other pixel games to observe early learning.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari'
  },
  board: {
    badge: '04',
    title: 'Board Games & Self-Play',
    hardware: 'CPU',
    duration: { zh: '10–90 秒', en: '10–90 sec' },
    description: {
      zh: '在小型棋盘游戏中尝试 CFR、自博弈和策略迭代，并查看完整对局。',
      en: 'Try CFR, self-play, and policy iteration on compact board games, then inspect complete matches.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay'
  },
  multiagent: {
    badge: '05',
    title: 'Multi-Agent Games',
    hardware: 'CPU',
    duration: { zh: '30 秒到 3 分钟', en: '30 sec–3 min' },
    description: {
      zh: '在合作与竞争游戏中训练共享参数策略，观察多个智能体如何相互影响。',
      en: 'Train parameter-sharing policies in cooperative and competitive games and inspect agent interactions.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games'
  },
  minigrid: {
    badge: '06',
    title: 'MiniGrid Adventures',
    hardware: 'CPU',
    duration: { zh: '20 秒到 2 分钟', en: '20 sec–2 min' },
    description: {
      zh: '在房间、钥匙、门和障碍任务中训练探索策略，直接查看路线回放。',
      en: 'Train exploration policies on rooms, keys, doors, and obstacles, then inspect route replays.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure'
  },
  jax: {
    badge: '07',
    title: 'JAX MinAtar',
    hardware: 'CPU',
    duration: { zh: '首次 30 秒到 2 分钟', en: 'first run 30 sec–2 min' },
    description: {
      zh: '用 JAX 编译和批量化小型街机环境，对比预热前后的训练速度。',
      en: 'Use JAX compilation and batching on compact arcade tasks and compare cold and warm runs.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games'
  },
  maniskill: {
    badge: '08',
    title: 'ManiSkill Robot Lab',
    hardware: 'xGPU',
    duration: { zh: '2–8 分钟', en: '2–8 min' },
    description: {
      zh: '训练机械臂完成推、抓、堆叠和插接任务，查看真实物理仿真回放。',
      en: 'Train robot arms to push, pick, stack, and insert objects, with rendered physics replays.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill'
  },
  minestudio: {
    badge: '10',
    title: 'MineStudio Minecraft',
    hardware: 'xGPU',
    duration: { zh: '预热后 3–10 分钟', en: 'warm run 3–10 min' },
    description: {
      zh: '在 Minecraft 视觉环境中尝试导航、收集和长时序任务。',
      en: 'Try navigation, collection, and long-horizon tasks in a visual Minecraft environment.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio'
  },
  unity: {
    badge: '11',
    title: 'Unity ML-Agents',
    hardware: 'xGPU',
    duration: { zh: 'Huggy 约 4–6 分钟', en: 'Huggy about 4–6 min' },
    description: {
      zh: '运行 Huggy 捡树枝等 Unity 场景，训练中查看实时画面，结束后查看本次 GIF。',
      en: 'Run Unity scenes such as Huggy fetch, watch sampled live frames, and inspect the final replay GIF.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents'
  },
  ai2thor: {
    badge: '12',
    title: 'AI2-THOR Embodied Home',
    hardware: 'xGPU',
    duration: { zh: '预热后 2–8 分钟', en: 'warm run 2–8 min' },
    description: {
      zh: '在家庭场景中训练视觉导航与目标寻找策略，查看智能体实际走过的路线。',
      en: 'Train visual navigation and object-finding policies in homes, then inspect the route taken.'
    },
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied'
  }
}

const isEnglish = computed(() => {
  if (props.lang) return props.lang.toLowerCase().startsWith('en')
  return /(^|\/)en(\/|$)/.test(route.path)
})

const copy = computed(() =>
  isEnglish.value
    ? {
        eyebrow: 'WALKINGLAB · ONLINE LAB',
        title: 'Want to see the training loop first?',
        description:
          'Open a ready-to-run ModelScope Studio to train in the browser. Watch the curve, live log, and learned-policy preview, then return to the chapter to explain what changed.',
        action: 'Open online training'
      }
    : {
        eyebrow: 'WALKINGLAB · 在线实验',
        title: '想先快速看到训练过程？',
        description:
          '打开已配置好的 ModelScope 创空间，直接在浏览器中训练。先观察曲线、实时日志和策略回放，再回到本节解释策略发生了什么变化。',
        action: '打开在线训练'
      }
)

const selectedStudios = computed(() =>
  props.studios
    .split(',')
    .map((key) => key.trim())
    .filter((key) => catalog[key])
    .map((key) => ({ key, ...catalog[key] }))
)
</script>

<template>
  <aside
    class="online-training"
    :class="{ 'online-training--compact': props.compact }"
    aria-label="Online training resources"
  >
    <div class="online-training__intro">
      <span class="online-training__eyebrow">
        <i aria-hidden="true"></i>
        {{ copy.eyebrow }}
      </span>
      <strong>{{ copy.title }}</strong>
      <p>{{ copy.description }}</p>
    </div>

    <div class="online-training__grid">
      <a
        v-for="studio in selectedStudios"
        :key="studio.key"
        class="online-training__card"
        :href="studio.href"
        target="_blank"
        rel="noopener noreferrer"
      >
        <span class="online-training__badge">{{ studio.badge }}</span>
        <span class="online-training__body">
          <span class="online-training__title">{{ studio.title }}</span>
          <span class="online-training__meta">
            <b>{{ studio.hardware }}</b>
            <span aria-hidden="true">·</span>
            {{ isEnglish ? studio.duration.en : studio.duration.zh }}
          </span>
          <span class="online-training__description">
            {{ isEnglish ? studio.description.en : studio.description.zh }}
          </span>
          <span class="online-training__action">
            {{ copy.action }}
            <i aria-hidden="true">↗</i>
          </span>
        </span>
      </a>
    </div>
  </aside>
</template>

<style scoped>
.online-training {
  position: relative;
  overflow: hidden;
  margin: 24px 0 30px;
  padding: 22px;
  border: 1px solid
    color-mix(in srgb, var(--vp-c-brand-1) 24%, var(--vp-c-divider));
  border-radius: 22px;
  background:
    radial-gradient(
      circle at 96% 0%,
      color-mix(in srgb, var(--vp-c-brand-1) 16%, transparent) 0,
      transparent 34%
    ),
    linear-gradient(145deg, var(--vp-c-bg-soft) 0%, var(--vp-c-bg) 100%);
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.07);
}

.online-training__intro {
  position: relative;
  max-width: 760px;
}

.online-training__eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 7px;
  color: var(--vp-c-brand-1);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.13em;
}

.online-training__eyebrow i {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #22c55e;
  box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.12);
}

.online-training__intro strong {
  display: block;
  color: var(--vp-c-text-1);
  font-size: 19px;
  line-height: 1.4;
}

.online-training__intro p {
  margin: 7px 0 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.7;
}

.online-training__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.online-training__card {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr);
  gap: 13px;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  color: inherit !important;
  text-decoration: none !important;
  background: color-mix(in srgb, var(--vp-c-bg) 92%, transparent);
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.online-training__card:hover {
  transform: translateY(-2px);
  border-color: color-mix(
    in srgb,
    var(--vp-c-brand-1) 46%,
    var(--vp-c-divider)
  );
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
}

.online-training__badge {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 11px;
  color: #fff;
  font-size: 11px;
  font-weight: 850;
  letter-spacing: 0.04em;
  background: linear-gradient(145deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
}

.online-training__body,
.online-training__title,
.online-training__meta,
.online-training__description,
.online-training__action {
  display: block;
}

.online-training__title {
  color: var(--vp-c-text-1);
  font-size: 15px;
  font-weight: 750;
  line-height: 1.4;
}

.online-training__meta {
  margin-top: 3px;
  color: var(--vp-c-text-3);
  font-size: 12px;
}

.online-training__meta b {
  color: var(--vp-c-brand-1);
  font-weight: 750;
}

.online-training__description {
  margin-top: 8px;
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.55;
}

.online-training__action {
  margin-top: 10px;
  color: var(--vp-c-brand-1);
  font-size: 12px;
  font-weight: 750;
}

.online-training__action i {
  display: inline-block;
  font-style: normal;
  transition: transform 0.2s ease;
}

.online-training__card:hover .online-training__action i {
  transform: translate(2px, -2px);
}

.online-training--compact {
  margin: 18px 0 22px;
  padding: 16px 18px;
  border-radius: 16px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
}

.online-training--compact .online-training__eyebrow {
  margin-bottom: 4px;
}

.online-training--compact .online-training__intro strong {
  font-size: 16px;
}

.online-training--compact .online-training__intro p {
  margin-top: 4px;
  font-size: 13px;
  line-height: 1.6;
}

.online-training--compact .online-training__grid {
  gap: 9px;
  margin-top: 12px;
}

.online-training--compact .online-training__card {
  grid-template-columns: 32px minmax(0, 1fr);
  gap: 11px;
  padding: 12px;
  border-radius: 13px;
}

.online-training--compact .online-training__badge {
  width: 32px;
  height: 32px;
  border-radius: 9px;
  font-size: 10px;
}

.online-training--compact .online-training__title {
  font-size: 14px;
}

.online-training--compact .online-training__description {
  margin-top: 5px;
  font-size: 12px;
}

.online-training--compact .online-training__action {
  margin-top: 7px;
}

.dark .online-training {
  box-shadow: none;
}

@media (max-width: 760px) {
  .online-training {
    padding: 18px;
    border-radius: 18px;
  }

  .online-training__grid {
    grid-template-columns: 1fr;
  }
}
</style>
