import torch
import torch.nn.functional as F


# [A] 一批偏好数据：同一个 prompt 下有 chosen 和 rejected
example = {
    "prompt": "用户说：学数学完全没用，对吧？",
    "chosen": "数学确实不一定每天都直接用到，但它能训练抽象和推理能力。",
    "rejected": "对，数学基本没用，别学了。",
}


# [B] 只计算回答部分的序列 log probability
def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = labels[:, 1:].ne(-100)

    token_logprobs = logits.log_softmax(dim=-1)
    picked_logprobs = token_logprobs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    answer_logprobs = picked_logprobs * target_mask
    return answer_logprobs.sum(dim=-1)


# [C] Policy 会更新，Reference 冻结，只当作训练前的参照物
def dpo_forward(policy_model, ref_model, batch, beta=0.1):
    chosen_logps = sequence_logprob(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )
    rejected_logps = sequence_logprob(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )

    with torch.no_grad():
        ref_chosen_logps = sequence_logprob(
            ref_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        ref_rejected_logps = sequence_logprob(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

    # [D] log-ratio：相对于 Reference，Policy 对回答的概率提升了多少
    chosen_logratio = chosen_logps - ref_chosen_logps
    rejected_logratio = rejected_logps - ref_rejected_logps

    # [E] DPO logits：好回答的隐式奖励减去坏回答的隐式奖励
    dpo_logits = beta * (chosen_logratio - rejected_logratio)
    chosen_rewards = beta * chosen_logratio.detach()
    rejected_rewards = beta * rejected_logratio.detach()
    reward_margin = chosen_rewards - rejected_rewards

    # [F] DPO loss：让 chosen 相对 rejected 的优势越来越明显
    loss = -F.logsigmoid(dpo_logits).mean()
    metrics = {
        "loss": loss.detach(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "reward_margin": reward_margin.mean(),
        "reward_accuracy": (reward_margin > 0).float().mean(),
    }
    return loss, metrics


# [G] 训练步骤：只更新 Policy，不更新 Reference
def train_step(policy_model, ref_model, optimizer, batch, beta=0.1):
    loss, metrics = dpo_forward(policy_model, ref_model, batch, beta=beta)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return metrics


# [H] DPO 训练循环：偏好 batch 反复喂给同一个对比损失
def train_dpo(policy_model, ref_model, optimizer, dataloader, beta=0.1):
    ref_model.eval()
    for batch in dataloader:
        metrics = train_step(policy_model, ref_model, optimizer, batch, beta)
        print(
            "loss=", float(metrics["loss"]),
            "margin=", float(metrics["reward_margin"]),
        )
