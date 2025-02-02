import re


text = '## Task Description and High-Level Movements\n\nThe task is to pick up a black bowl located on a wooden cabinet and place it on a plate situated on a wooden table.  This involves a sequence of actions: approaching the cabinet, grasping the bowl, lifting the bowl, moving to the table, and placing the bowl on the plate.  The robot must navigate around potential obstacles and maintain a stable grip on the bowl throughout the process.\n\n\nBased on the trajectory, we can identify the following high-level movements:\n\n1. **Approaching the Cabinet:** Steps 0-12. This involves moving the robotic arm upwards and forwards, adjusting its orientation to approach the cabinet and position itself for grasping the bowl.\n\n2. **Grasping the Bowl:** Steps 13-44.  This phase encompasses moving the arm to the bowl\'s location, orienting the gripper appropriately, and securely grasping the bowl.\n\n3. **Lifting the Bowl:** Steps 45-58. This involves lifting the bowl from the cabinet to a safe height, ensuring it remains securely held within the gripper.\n\n4. **Moving to the Plate:** Steps 59-124.  This requires moving the arm horizontally towards the plate while keeping the bowl steady and avoiding obstacles.\n\n5. **Placing the Bowl:** Steps 125-138. This entails moving the arm to the plate\'s location, carefully lowering and releasing the bowl onto the plate.\n\n**Justifications:**\n\n* **Approaching the Cabinet:** This is necessary to get within reach of the bowl before attempting to grasp it.  The robot needs to navigate towards the cabinet, potentially avoiding obstacles.\n* **Grasping the Bowl:**  This is the essential step for successfully completing the task; the robot must accurately position itself and the gripper to secure a grip on the bowl.\n* **Lifting the Bowl:**  Lifting is crucial to clear the bowl from the cabinet and initiate transport to the plate.\n* **Moving to the Plate:** This movement transports the bowl from its starting location to the desired location. The robot needs to adjust its trajectory based on obstacles and distance to ensure the bowl remains stable.\n* **Placing the Bowl:** This final step releases the bowl securely onto the plate, achieving the goal of the task.\n\n\n## Step-by-Step Reasoning\n\n```python\nreasoning = {\n    0: "<task>Pick up the black bowl and place it on the plate.<plan>Approach cabinet, grasp bowl, lift bowl, move to plate, place bowl.<subtask>Approach cabinet.<subtask_reason>The robot needs to get close to the bowl before grasping it. Initial upward movement clears any obstacles.<move>move up<move_reason>Initial upward movement provides clearance and prepares for forward movement towards the cabinet.",\n    1: "<task>Pick up the black bowl and place it on the plate.<plan>Approach cabinet, grasp bowl, lift bowl, move to plate, place bowl.<subtask>Approach cabinet.<subtask_reason>Moving forward and up allows the robot to approach the cabinet while avoiding potential obstacles.<move>move forward up<move_reason>Continuing the approach toward'
tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")


# Start with a base pattern for the numbers
pattern = r"(\d+):\s*\"?"

# Append tag patterns for each tag in the list
for tag in tags:
    pattern += r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"
    print(pattern)

# Apply the regex pattern
matches = re.findall(pattern, text)

if len(matches) == 0:
    pattern = r"(\d+):\s*\"?"
    for tag in tags:
        pattern += r"\s*<" + tag + r">([^<]+)"
        print(pattern)
    
# Print matches
print("matches:", matches)