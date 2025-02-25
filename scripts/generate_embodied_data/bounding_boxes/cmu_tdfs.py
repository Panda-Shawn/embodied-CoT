import tensorflow_datasets as tfds

def print_structure(data, indent=0):
    """
    递归打印数据集的结构（字典键或列表属性）。
    
    参数:
        data: 数据集中的样本（可能是字典、列表或其他类型）。
        indent: 缩进级别，用于格式化输出。
    """
    if isinstance(data, dict):
        for key, value in data.items():
            print(" " * indent + f"Key: {key} (Type: {type(value).__name__})")
            print_structure(value, indent + 4)  # 递归处理嵌套字典
    elif isinstance(data, (list, tuple)):
        print(" " * indent + f"List/Tuple (Length: {len(data)})")
        if len(data) > 0:
            print_structure(data[0], indent + 4)  # 递归处理列表中的第一个元素
    else:
        print(" " * indent + f"Value: {data} (Type: {type(data).__name__})")

# 加载数据集
# ds = tfds.load("cmu_franka_exploration_dataset_converted_externally_to_rlds", 
            #    data_dir="/data/lwh/", 
            #    split=f"train[{0}%:{100}%]")
# ds = tfds.load("libero_10_no_noops", data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
ds = tfds.load("bridge_orig", data_dir="/data/lzx/bridge_dataset", split=f"train[{0}%:{100}%]")
# 获取数据集中的一个样本
sample = next(iter(ds))

# 打印样本的结构
print("Dataset sample structure:")
print_structure(sample)