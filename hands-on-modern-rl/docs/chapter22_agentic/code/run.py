# run.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import SandboxEnv
from policy import Policy
from trainer import GRPOAgentTrainer

# 加载一个小模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化各组件
env = SandboxEnv(timeout=10)
policy = Policy(model, tokenizer, lr=5e-5)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
policy.set_ref_model(ref_model)


TASK_EXPECTED_OUTPUTS = {
    "写 Python 代码计算 F(10)，规定 F(0)=0、F(1)=1，并且只输出结果。": "55",
    "写 Python 代码判断字符串 'racecar' 是否为回文，并且只输出 True 或 False。": "True",
    "写 Python 代码把 [5, 1, 4, 2, 8] 从小到大排序，并且只输出排序后的列表。": "[1, 2, 4, 5, 8]",
}


# 定义可验证 reward：最后一行必须等于该题的期望输出
def code_reward(trajectory):
    """如果某次代码执行的最后一行等于期望输出，reward = 1。"""
    expected = TASK_EXPECTED_OUTPUTS[trajectory["prompt"]]
    for interaction in trajectory["interactions"]:
        obs = interaction.get("observation", "")
        output_lines = [line.strip() for line in obs.splitlines() if line.strip()]
        if output_lines and output_lines[-1] == expected:
            return 1.0
    return 0.0


# 训练 prompts
prompts = list(TASK_EXPECTED_OUTPUTS)

# 开始训练
trainer = GRPOAgentTrainer(
    policy=policy,
    env=env,
    reward_fn=code_reward,
    group_size=4,
    max_turns=3,
)
history = trainer.fit(prompts, n_steps=30)
