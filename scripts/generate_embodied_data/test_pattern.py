import re


text = '\n134: "<task>Pick up the black bowl from the top drawer of the wooden cabinet and place it on the plate.</task><plan>Retreat.</plan><subtask>Retreat</subtask><subtask_reason>The robot is moving away. It needs to tilt down.</subtask_reason><move>tilt down</move><move_reason>Tilts down to a neutral position.</move_reason>"'
tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")


pattern = r"(\d+):\s*\"?"
for tag in tags:
    pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

# 动态生成匹配模式
# pattern = r""
# for tag in tags:
#     pattern += rf"<{tag}>(.*?)</{tag}>"

matches = re.findall(pattern, text)

print("matches:", matches)