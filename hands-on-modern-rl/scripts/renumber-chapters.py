#!/usr/bin/env python3
"""Renumber chapters 7-32 down to 5-30 to close the gap left by bandits merge."""

import os
import re

CHAPTER_MAP = {old: old - 2 for old in range(7, 33)}
# 7->5, 8->6, ..., 32->30

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')


def shift_num(n):
    """Return new chapter number for old number n, or None if n < 7."""
    return CHAPTER_MAP.get(n)


def replace_chapter_ref(match):
    """Regex callback: 第 X 章 -> 第 NEW 章 (atomic)."""
    old = int(match.group(1))
    new = shift_num(old)
    if new is None:
        return match.group(0)
    return f'第 {new} 章'


def replace_section_ref(match):
    """Regex callback: X.Y -> NEW.Y in H1 only."""
    old = int(match.group(1))
    new = shift_num(old)
    if new is None:
        return match.group(0)
    return f'# {new}.{match.group(2)}'


def update_file(filepath):
    """Update H1 and 第 X 章 references in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 1) Update H1 patterns: "# 第 X 章" and "# X.Y"
    def h1_chap(match):
        old = int(match.group(1))
        new = shift_num(old)
        if new is None:
            return match.group(0)
        return f'# 第 {new} 章'

    content = re.sub(r'^# 第 (\d+) 章', h1_chap, content, count=1, flags=re.MULTILINE)

    def h1_sec(match):
        old = int(match.group(1))
        new = shift_num(old)
        if new is None:
            return match.group(0)
        return f'# {new}.{match.group(2)}'

    content = re.sub(r'^# (\d+)\.(\d+)', h1_sec, content, count=1, flags=re.MULTILINE)

    # 2) Update 第 X 章 references everywhere in body
    content = re.sub(r'第 (\d+) 章', replace_chapter_ref, content)

    # 3) Update H2/H3 section refs like "## X.Y ..." (rare but exists)
    # Skip - too risky

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    changed = 0
    scanned = 0
    for root, dirs, files in os.walk(DOCS_DIR):
        # Skip archive dirs
        if '_archive' in root or '.vitepress' in root:
            continue
        for fname in files:
            if not fname.endswith('.md'):
                continue
            filepath = os.path.join(root, fname)
            scanned += 1
            if update_file(filepath):
                changed += 1
                print(f'  updated: {os.path.relpath(filepath, DOCS_DIR)}')
    print(f'\nScanned {scanned} files, updated {changed}.')


if __name__ == '__main__':
    main()
