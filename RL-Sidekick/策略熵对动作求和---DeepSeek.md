# 策略熵对动作求和 - DeepSeek

> 由 RL Sidekick 自动记录 · 源页面：https://chat.deepseek.com/a/chat/s/b495f13d-e741-4f3c-952e-96db50d2f4ee

### 2026-11-12 12:10 · 💡 灵感

> 🧠 我的想法：“batch 是一次参数更新样本，是 rollout 数据”
✅ 对，但更准确：mini-batch 是从 rollout 里切出来的一次参数更新用的 64 个 transition。

“rollout 是 2048 的意思是我理解是 2048 次环境交互产生的数据”
✅ 单环境下对；多环境下是 2048 × n_envs 次交互。

---

