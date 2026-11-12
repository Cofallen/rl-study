# policy.py
import torch
import torch.nn.functional as F


class Policy:
    """包装一个语言模型，提供 generate() 和 train_step_with_advantage() 两个接口。"""

    def __init__(self, model, tokenizer, lr=1e-5, clip_eps=0.2, kl_coef=0.04):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.ref_model = None  # reference model for KL penalty

    def set_ref_model(self, ref_model):
        """保存一份初始权重的拷贝，用作 KL 散度计算的锚点。"""
        self.ref_model = ref_model.to(self.model.device).eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        """推理模式：给定 prompt，生成文本。"""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
        prompt_width = inputs["input_ids"].shape[1]
        completion_ids = outputs[0, prompt_width:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def _token_logprobs(self, model, prompt: str, response: str) -> torch.Tensor:
        """保留回答中每个 token 的 log probability，并保留梯度。"""
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        response_inputs = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"].to(model.device)
        response_ids = response_inputs["input_ids"].to(model.device)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        prompt_width = prompt_ids.shape[1]
        response_logits = logits[:, prompt_width - 1 : -1, :]
        logprobs = F.log_softmax(response_logits, dim=-1)
        return logprobs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def get_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        """无梯度地记录采样策略对回答中每个 token 的 log probability。"""
        return self._token_logprobs(self.model, prompt, response)

    @torch.no_grad()
    def _get_ref_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        """计算参考策略对回答中每个 token 的 log probability。"""
        return self._token_logprobs(self.ref_model, prompt, response)

    def train_step_with_advantage(self, trajectories: list):
        """
        一个原始 GRPO 训练步。
        trajectories: list of ([(turn_prompt, turn_response), ...], advantage)

        每条多轮轨迹只有一个结果优势；ratio、clip、KL 逐 token 计算。
        环境 observation 只进入下一轮 prompt，不进入动作 token 的 loss。
        """
        self.model.train()
        self.optimizer.zero_grad()
        trajectory_losses = []

        for turns, advantage in trajectories:
            turn_token_losses = []
            for prompt, response in turns:
                new_logprobs = self._token_logprobs(self.model, prompt, response)
                if new_logprobs.numel() == 0:
                    continue

                # 每批轨迹只更新一次，因此采样策略就是本次前向的 detach 版本。
                old_logprobs = new_logprobs.detach()
                ratio = torch.exp(new_logprobs - old_logprobs)
                advantage_tensor = new_logprobs.new_tensor(advantage)
                unclipped = ratio * advantage_tensor
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantage_tensor

                if self.ref_model is not None:
                    ref_logprobs = self._get_ref_logprobs(prompt, response)
                    log_ratio_ref = ref_logprobs - new_logprobs
                    per_token_kl = (
                        torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                    )
                else:
                    per_token_kl = torch.zeros_like(new_logprobs)

                per_token_loss = (
                    -torch.minimum(unclipped, clipped)
                    + self.kl_coef * per_token_kl
                )
                turn_token_losses.append(per_token_loss.reshape(-1))

            if turn_token_losses:
                # 每条轨迹先按自己的动作 token 数归一化，再对轨迹求平均。
                trajectory_losses.append(torch.cat(turn_token_losses).mean())

        if not trajectory_losses:
            return 0.0

        total_loss = torch.stack(trajectory_losses).mean()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
