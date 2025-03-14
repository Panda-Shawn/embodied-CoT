import json

d = json.load(open('/data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_goal_w_mask/cot/libero_goal_primitives.json', 'r'))

filtered_primitives = []
prev_primitive = None
for i, primitive in enumerate(d["libero_goal_Task_7_Demo_45"]):
    if primitive != prev_primitive:
        filtered_primitives.append((len(filtered_primitives), i, primitive))
        prev_primitive = primitive

print(len(filtered_primitives))
print(filtered_primitives)