import re

with open('docs/appendix_game_projects/intro.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Header and intro
header_match = re.search(r'^(# 附录 D：学习资料与项目推荐\n\n> \*\*本附录目标\*\*：.*?)\n\n## 项目推荐', content, re.DOTALL)
if not header_match:
    print("Header not found")
header = header_match.group(1)

# 2. 项目推荐 (Milestones)
projects_match = re.search(r'## 项目推荐\n(.*?)(\n## RL 环境与工具|\Z)', content, re.DOTALL)
projects = projects_match.group(1)

# 3. RL 环境与工具
env_match = re.search(r'## RL 环境与工具\n(.*?)(\n## 学习资料推荐|\Z)', content, re.DOTALL)
env = env_match.group(1)

# 4. 学习资料推荐
learning_match = re.search(r'## 学习资料推荐\n(.*)', content, re.DOTALL)
learning = learning_match.group(1)

# Constructing new content
new_content = """# 附录 D：学习资料与复现项目推荐

> **本附录目标**：为读者的后续进阶提供清晰的导航。前半部分整理了理论扎实、深入浅出的教材与课程资源，帮助你系统性夯实基础或钻研前沿；后半部分则梳理了强化学习在游戏与仿真生态中的经典里程碑与常用环境，为你寻找下一个动手复现的实战项目提供灵感与坐标系。

## 学习资料推荐
""" + learning + """

## 复现项目推荐

强化学习的发展往往伴随着极具挑战性的游戏与仿真任务。从经典的棋盘游戏到复杂的 3D 物理仿真，再到多智能体博弈，这些环境不仅是检验算法的试金石，也是初学者动手复现的最佳练兵场。以下按类型梳理了本课程涉及的常用仿真环境，以及 30 个里程碑式的 RL 项目，为你寻找实战方向提供参考。

### 常用 RL 环境与工具
""" + env + """
### 经典里程碑项目参考
""" + projects

with open('docs/appendix_game_projects/intro.md', 'w', encoding='utf-8') as f:
    f.write(new_content)
