#!/usr/bin/env python3
"""
Mini Deep Research RL using GRPO and a small LLM (e.g., Qwen2.5-0.5B-Instruct).
This script demonstrates how to train a research agent with GRPO in a single-turn
format by simulating the search environment inside the reward functions.

Prerequisites:
    pip install -r requirements.txt
"""

import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

import sys
import os

# Import the offline environment from the docs directory script
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../docs/chapter22_agentic/code",
    )
)
from deep_research_rl_benchmark import DOCS, TASKS, search, Document

# 1. Prepare the Dataset
def build_dataset():
    # Build the knowledge base context (Titles and IDs only, not the full text)
    kb_context = "Available Documents:\n"
    for doc in DOCS:
        kb_context += f"- [{doc.doc_id}] {doc.title}\n"
    
    prompts = []
    for task in TASKS:
        # We give the model the question and the available document titles.
        # It must generate the query, select the doc, and answer.
        prompt = f"""You are a Deep Research Agent. Your task is to answer the user's question by searching for the correct document.
{kb_context}

Question: {task.question}

You must output your reasoning, the search query you would use to find the document, the exact doc_id you choose to open, and the final short answer.
Use this exact format:
<think>
...
</think>
<query>...</query>
<doc>...</doc>
<answer>...</answer>
"""
        prompts.append({
            "prompt": [{"role": "user", "content": prompt}],
            "gold_answer": task.gold_answer,
            "support_doc_id": task.support_doc_id
        })
    return Dataset.from_list(prompts)

# 2. Reward Functions
def extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def completion_to_text(completion) -> str:
    """Normalize TRL string and conversational completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward 1.0 if all required XML tags are present."""
    rewards = []
    for comp in completions:
        text = completion_to_text(comp)
        has_think = "<think>" in text and "</think>" in text
        has_query = "<query>" in text and "</query>" in text
        has_doc = "<doc>" in text and "</doc>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        if has_think and has_query and has_doc and has_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def search_validity_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward 1.0 if the chosen <doc> is actually returned in the top-4 results 
    for the generated <query>. This forces the model to write realistic queries.
    """
    rewards = []
    import random
    rng = random.Random(42) # Deterministic for reward eval
    for comp in completions:
        text = completion_to_text(comp)
        query = extract_xml_tag(text, "query")
        doc_id = extract_xml_tag(text, "doc")
        if not query or not doc_id:
            rewards.append(0.0)
            continue
            
        # Simulate the search environment
        results = search(query, rng, limit=4)
        result_ids = [d.doc_id for d in results]
        
        if doc_id in result_ids:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def citation_reward_func(completions, support_doc_id, **kwargs) -> list[float]:
    """Reward 1.0 if the model cites the correct document."""
    rewards = []
    for comp, gold_doc in zip(completions, support_doc_id):
        doc_id = extract_xml_tag(completion_to_text(comp), "doc")
        if doc_id == gold_doc:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, gold_answer, **kwargs) -> list[float]:
    """Reward 1.0 if the final answer is correct."""
    rewards = []
    for comp, gold_ans in zip(completions, gold_answer):
        ans = extract_xml_tag(completion_to_text(comp), "answer")
        if ans.lower() == gold_ans.lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 3. Main Training Script
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Mini DeepResearch Agent with GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="The base model to train")
    parser.add_argument("--output_dir", type=str, default="./mini-deepresearch-grpo", help="Output directory")
    args = parser.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Optional: If you want to run this locally on Mac (MPS) or small GPU
    # adjust the parameters below.
    paper_grpo_options = {
        "beta": 0.04,
        "epsilon": 0.2,
        "num_iterations": 1,
        "scale_rewards": "group",
        "importance_sampling_level": "token",
        "loss_type": "grpo",
    }
    # TRL 0.24 uses the paper's KL estimator directly. Newer releases can
    # bias-correct it, so disable that extension when the option is available.
    if "use_bias_correction_kl" in GRPOConfig.__dataclass_fields__:
        paper_grpo_options["use_bias_correction_kl"] = False

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-5,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        # GRPO specific
        num_generations=4,  # Number of rollouts per prompt
        max_completion_length=200,
        # Pin the original objective instead of inheriting later TRL defaults.
        **paper_grpo_options,
        remove_unused_columns=False,  # Required for custom kwargs in reward funcs
        logging_steps=1,
        report_to="none",
    )

    dataset = build_dataset()

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[
            format_reward_func,
            search_validity_reward_func,
            citation_reward_func,
            accuracy_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO Training for {model_name}...")
    trainer.train()
    
    # Save the final model
    final_output = f"{args.output_dir}-final"
    trainer.save_model(final_output)
    print(f"Training complete. Model saved to {final_output}")

if __name__ == "__main__":
    main()
