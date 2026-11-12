#!/usr/bin/env python3
"""
Generate synthetic Deep Research SFT data using an OpenAI-compatible API.

This script simulates a simplified "Multi-Agent" workflow (Planner -> Searcher -> Summarizer)
using a single powerful LLM via API (e.g., GPT-4o, DeepSeek-V3, or Qwen-Max) to generate
high-quality reasoning trajectories for our offline benchmark tasks.

Prerequisites:
    pip install openai
"""

import os
import json
from openai import OpenAI
import sys

# Import our offline environment from the benchmark script
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../docs/chapter22_agentic/code",
    )
)
from deep_research_rl_benchmark import DOCS, TASKS

# Initialize OpenAI Client (Replace with your compatible API base and key)
# Example for DeepSeek API: base_url="https://api.deepseek.com/v1"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

def generate_trajectory(task):
    """
    Simulate a multi-step thought process to generate a high-quality SFT trajectory.
    We instruct the API to output the exact XML format required by our GRPO script.
    """
    
    # 1. Build the Knowledge Base context
    kb_context = "Available Documents:\n"
    for doc in DOCS:
        kb_context += f"- [{doc.doc_id}] {doc.title}\n"
    
    # 2. System Prompt enforcing the Multi-Agent "Thought Process"
    system_prompt = """You are an elite AI Research Team.
You must solve the user's question by selecting the exact document from the Available Documents list.

To do this perfectly, you must simulate this process in your <think> tag:
1. PLAN: What are the key entities in the question?
2. SEARCH: What exact query string will match the target document's title or content?
3. SELECT: Which doc_id exactly matches?
4. SUMMARIZE: What is the final concise answer?

You MUST output exactly in this format:
<think>
[Your simulated team reasoning here]
</think>
<query>[Your search query here]</query>
<doc>[The chosen doc_id here]</doc>
<answer>[Your final short answer here]</answer>
"""

    user_prompt = f"{kb_context}\nQuestion: {task.question}\nGold Answer (for your reference to ensure correctness): {task.gold_answer}\nTarget Doc ID: {task.support_doc_id}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    print(f"Generating synthetic data using {MODEL_NAME}...")
    synthetic_data = []
    
    for i, task in enumerate(TASKS):
        print(f"Processing Task {i+1}/{len(TASKS)}: {task.question}")
        trajectory = generate_trajectory(task)
        
        if trajectory:
            synthetic_data.append({
                "task_id": task.task_id,
                "question": task.question,
                "gold_answer": task.gold_answer,
                "support_doc_id": task.support_doc_id,
                "synthetic_trajectory": trajectory
            })
            print("Successfully generated trajectory.\n")
        else:
            print("Failed to generate trajectory.\n")
            
    # Save to JSONL for SFT training later
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_sft_data.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Done! Saved {len(synthetic_data)} trajectories to {output_file}")

if __name__ == "__main__":
    main()
