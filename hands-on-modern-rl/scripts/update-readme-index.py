#!/usr/bin/env python3
"""Update README.md and README.zh.md chapter/appendix tables to new numbering."""

import re

# Mapping of OLD chapter numbers (as shown in README) to NEW chapter numbers
# README uses 2-digit format like "01", "02"
ZH_TO_NEW = {
    '01': '1',     # CartPole
    # '02': removed (bandits merged)
    '03': '2',     # MDP/RL basic definitions
    # 04-06 don't exist
    '07': '5',     # DQN
    '08': '6',     # Policy Gradient
    '09': '7',     # Actor-Critic
    '10': '8',     # PPO
    '11': '9',     # Continuous Control
    '12': '10',    # Offline RL
    '13': '11',    # Imitation/Meta
    '14': '12',    # Exploration/MARL
    '15': '13',    # RLHF
    '16': '14',    # LLM RL Industrial
    '17': '15',    # DPO
    '18': '16',    # GRPO
    '19': '17',    # Reasoning
    '20': '18',    # PRM
    '21': '19',    # CAI
    '22': '20',    # Agentic
    '23': '21',    # SWE
    '24': '22',    # Deep Research
    '25': '23',    # Computer Use
    '26': '24',    # VLM
    '27': '25',    # Audio
    '28': '26',    # VLA
    '29': '27',    # Visual Generation
    '30': '28',    # Alignment Failures
    # '31': removed (AlphaEvolve merged)
    '32': '29',    # Self-Play
}


def renumber_zh(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    # Pattern: line starts with "| DD   |" or "| DD " where DD is 2-digit chapter
    # Replace with new number
    def replace_chap(match):
        old = match.group(1)
        new = ZH_TO_NEW.get(old)
        if new is None:
            return match.group(0)
        prefix = match.group(0)[:match.start(1) - match.start(0)]
        suffix = match.group(0)[match.end(1) - match.start(0):]
        return prefix + new + suffix

    # Match "| DD   " or "| DD " patterns at start of table rows
    # Pattern: pipe, whitespace, 2 digits, whitespace
    content = re.sub(r'(\| \s*)(\d{2})(\s+\|)', lambda m: m.group(1) + ZH_TO_NEW.get(m.group(2), m.group(2)) + m.group(3), content)

    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {path}')
    else:
        print(f'No changes to {path}')


if __name__ == '__main__':
    renumber_zh('/Users/sanbu/Code/2026重要开源项目/hands-on-modern-rl/README.zh.md')
    renumber_zh('/Users/sanbu/Code/2026重要开源项目/hands-on-modern-rl/README.md')
