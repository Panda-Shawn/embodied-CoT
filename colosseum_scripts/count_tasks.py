import os

# 设置文件夹路径
folder_path = "/home/nus/colosseum/colosseum_dataset_maps"

# 获取文件夹中所有文件
files = os.listdir(folder_path)

# 提取任务名（假设文件名格式为 "任务名_id_to_name.json"）
task_names = set()
for file in files:
    if file.endswith("_id_to_name.json"):
        task_name = file.rsplit("_id_to_name.json", 1)[0]  # 去掉后缀部分
        task_name = task_name.rsplit("_", 1)[0]  # 去掉后缀部分
        task_names.add(task_name)
    if file.endswith("_name_to_id.json"):
        task_name = file.rsplit("_name_to_id.json", 1)[0]  # 去掉后缀部分
        task_name = task_name.rsplit("_", 1)[0]  # 去掉后缀部分
        task_names.add(task_name)


# 统计任务数量
task_list = sorted(task_names)
task_count = len(task_list)

# 输出结果
print(f"共有 {task_count} 个任务:")
print(task_list)
