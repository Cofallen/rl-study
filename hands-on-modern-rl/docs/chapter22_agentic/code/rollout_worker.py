# rollout_worker.py

class RolloutWorker:
    """
    驱动 Agent Loop，收集多轮轨迹。
    轨迹结构: [(prompt, response_1, obs_1, response_2, obs_2, ..., final_response), reward]
    """

    def __init__(self, policy, env, max_turns=5):
        self.policy = policy
        self.env = env
        self.max_turns = max_turns

    def rollout(self, prompt: str, reward_fn) -> dict:
        """
        执行一次完整的 Agent Loop，返回轨迹和 reward。
        reward_fn: callable(trajectory) -> float
        """
        messages = [{"role": "user", "content": prompt}]
        trajectory = {"prompt": prompt, "interactions": []}

        for turn in range(self.max_turns):
            # 模型生成下一步动作
            context = self._format_context(messages)
            model_output = self.policy.generate(context)

            # 解析模型输出：判断是代码执行还是最终回答
            action = self._parse_action(model_output)

            if action["type"] == "finish":
                trajectory["interactions"].append({
                    "turn": turn,
                    "context": context,
                    "response": model_output,
                    "action": action,
                    "observation": None,
                })
                trajectory["final_response"] = action.get("answer", model_output)
                break

            # 环境执行动作
            obs = self.env.step(action["type"], action["args"])

            trajectory["interactions"].append({
                "turn": turn,
                "context": context,
                "response": model_output,
                "action": action,
                "observation": obs["observation"],
            })

            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": f"执行结果:\n{obs['observation']}"})

            if obs.get("done"):
                break

        # 计算奖励
        trajectory["reward"] = reward_fn(trajectory)
        return trajectory

    def _format_context(self, messages):
        """把多轮消息列表拼成模型能理解的 prompt。"""
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            else:
                parts.append(f"Assistant: {msg['content']}")
        return "\n".join(parts)

    def _parse_action(self, model_output: str) -> dict:
        """
        从模型输出中解析动作类型和参数。
        简化实现：按标记解析。
        """
        if "```python" in model_output:
            # 提取代码块
            code = model_output.split("```python")[1].split("```")[0]
            return {"type": "execute_code", "args": {"code": code}}
        elif "FINAL ANSWER:" in model_output:
            answer = model_output.split("FINAL ANSWER:")[1].strip()
            return {"type": "finish", "answer": answer}
        else:
            return {"type": "execute_code", "args": {"code": model_output}}
