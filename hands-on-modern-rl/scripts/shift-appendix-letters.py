#!/usr/bin/env python3
"""Shift appendix letters C-G to B-F in config.mjs."""

import re

CONFIG = '/Users/sanbu/Code/2026重要开源项目/hands-on-modern-rl/docs/.vitepress/config.mjs'

with open(CONFIG, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# Pattern: text: 'X.<anything>' where X is C, D, E, F, or G
# Replace X with previous letter
def shift(match):
    letter = match.group(1)
    new_letter = chr(ord(letter) - 1)
    return f"text: '{new_letter}.{match.group(2)}'"

content = re.sub(r"text: '([C-G])\.([^']+)'", shift, content)

if content != original:
    with open(CONFIG, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Shifted appendix letters')
else:
    print('No changes')
