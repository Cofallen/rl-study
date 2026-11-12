#!/usr/bin/env python3
"""Fix intro.md H1 numbers — they got double-shifted by the previous script.

The directory name is the source of truth:
  chapter09_actor_critic → chapter 7
  chapter10_ppo          → chapter 8
  ...
  chapterNN               → chapter NN - 2  (for NN in 7..32)
"""

import os
import re

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')


def expected_chapter(dirname):
    """chapter09_actor_critic -> 7, chapter32_selfplay -> 30, etc."""
    m = re.match(r'^chapter(\d{2})_', dirname)
    if not m:
        return None
    n = int(m.group(1))
    if n <= 6:
        return n  # chapters 1-6 unchanged
    return n - 2  # shift down by 2


def fix_file(filepath, expected):
    """Force the H1 to match expected chapter number."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    # Replace "# 第 X 章" with "# 第 EXPECTED 章" (only first occurrence)
    content = re.sub(rf'^# 第 \d+ 章', f'# 第 {expected} 章', content, count=1, flags=re.MULTILINE)

    # Replace "# X.Y" with "# EXPECTED.Y" (only first occurrence)
    def sub_sec(match):
        return f'# {expected}.{match.group(1)}'

    content = re.sub(r'^# \d+\.(\d+)', sub_sec, content, count=1, flags=re.MULTILINE)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    changed = 0
    for entry in sorted(os.listdir(DOCS_DIR)):
        dirpath = os.path.join(DOCS_DIR, entry)
        if not os.path.isdir(dirpath):
            continue
        expected = expected_chapter(entry)
        if expected is None:
            continue
        # Check all .md files in this directory (and one level deep)
        files_to_check = []
        for fname in os.listdir(dirpath):
            if fname.endswith('.md'):
                files_to_check.append(os.path.join(dirpath, fname))
        for sub in os.listdir(dirpath):
            subdir = os.path.join(dirpath, sub)
            if os.path.isdir(subdir):
                for fname in os.listdir(subdir):
                    if fname.endswith('.md'):
                        files_to_check.append(os.path.join(subdir, fname))

        for fp in files_to_check:
            if fix_file(fp, expected):
                changed += 1
                print(f'  fixed: {os.path.relpath(fp, DOCS_DIR)} -> ch {expected}')
    print(f'\nFixed {changed} files.')


if __name__ == '__main__':
    main()
