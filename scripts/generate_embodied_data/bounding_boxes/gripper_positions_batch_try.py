import cv2
import matplotlib
import mediapy
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline
import tensorflow_datasets as tfds
import json
from utils import NumpyFloatValuesEncoder
import time

import os



checkpoint = "google/owlvit-base-patch16"

# detector 自动使用cuda:0
# detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
device = 'cuda:2'
model.to(device)

sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


sam_model.to(device)

image_dims = (256, 256)
image_label = "image"
sam_batch_num = 12
owlvit_batch_num = 1

def get_bounding_boxes(img, prompt="the black robotic gripper"):

    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)
   
    return predictions

# def get_bounding_boxes_batch(images, prompt="the black robotic gripper"):
#     """
#     对 images 列表分组，对每个子列表调用 detector，最后拼接所有预测结果。

#     参数:
#         images (list): 包含多个图像的列表。
#         prompt (str): 检测目标的文本提示。
#         group_size (int): 每个子列表的大小（分组大小）。

#     返回:
#         list: 包含所有图像预测结果的列表。
#     """
#     # 初始化总的预测结果列表
#     all_predictions = []

#     # 对 images 列表分组
#     for i in range(0, len(images), owlvit_batch_num):
#         # 获取当前分组的子列表
#         group_images = images[i:i + owlvit_batch_num]
#         # group_images = [group_images]
#         # 对当前分组调用 detector
#         group_predictions = detector(
#             group_images,
#             candidate_labels=[[prompt] for _ in range(len(group_images))],  # 每个图像对应一个 prompt
#             threshold=0.01
#         )

#         # 将当前分组的预测结果添加到总列表中
#         all_predictions.extend(group_predictions)
        
#         print(f'count get bounding box from image {i} to {i + owlvit_batch_num}')

#     return all_predictions
def get_bounding_boxes_batch(images, prompt="the black robotic gripper"):
    """
    对 images 列表分组，对每个子列表调用 detector，最后拼接所有预测结果。

    参数:
        images (list): 包含多个图像的列表。
        prompt (str): 检测目标的文本提示。
        group_size (int): 每个子列表的大小（分组大小）。

    返回:
        list: 包含所有图像预测结果的列表。
    """
    # 初始化总的预测结果列表
    all_predictions = []

    # 对 images 列表分组
    for i in range(0, len(images), owlvit_batch_num):
        # 获取当前分组的子列表
        group_images = images[i:i + owlvit_batch_num]
        # group_images = [group_images]
        # 对当前分组调用 detector
        inputs = processor(text=[[prompt] for _ in range(len(group_images))], images=group_images, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)
        target_sizes = [x.size[::-1] for x in group_images]
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)


        # 将当前分组的预测结果添加到总列表中
        all_predictions.extend(results)
        
        print(f'count get bounding box from image {i} to {i + owlvit_batch_num}')

    return all_predictions

def show_box(box, ax, meta, color):
    x0, y0 = box["xmin"], box["ymin"]
    w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, "{:.3f}".format(meta["score"]), color="white")


def get_median(mask, p):
    row_sum = np.sum(mask, axis=1)
    cumulative_sum = np.cumsum(row_sum)

    if p >= 1.0:
        p = 1

    total_sum = np.sum(row_sum)
    threshold = p * total_sum

    return np.argmax(cumulative_sum >= threshold)


def get_gripper_mask(img, pred):
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]

   
    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt")
    
    # 将 inputs 移动到与模型相同的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}


    # 使用模型进行推理
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # 将输出数据移动到 CPU
    outputs.pred_masks = outputs.pred_masks.to("cpu")

    # 后处理掩码
    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0].numpy()

   
    return mask

def get_gripper_mask_batch(images, preds):

    # 提取所有图像的边界框
    boxes = [
        [
            round(pred["box"]["xmin"], 2),
            round(pred["box"]["ymin"], 2),
            round(pred["box"]["xmax"], 2),
            round(pred["box"]["ymax"], 2),
        ]
        for pred in preds
    ]

    # 将边界框组织成批量形式
    input_boxes = [[[box]] for box in boxes]

    # 批量处理输入图像和边界框
    inputs = sam_processor(images, input_boxes=input_boxes, return_tensors="pt")

    # 将 inputs 移动到与模型相同的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 使用模型进行批量推理
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # 将输出数据移动到 CPU
    outputs.pred_masks = outputs.pred_masks.to("cpu")

    # 后处理掩码（只取第一个图像的掩码）
    masks = []

    # 遍历每个图像的掩码
    for i in range(outputs.pred_masks.shape[0]):
        # 对每个图像的掩码进行后处理
        mask = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks[i].unsqueeze(0),  # 取第 i 个图像的掩码
            inputs["original_sizes"][i].unsqueeze(0),  # 第 i 个图像的原始尺寸
            inputs["reshaped_input_sizes"][i].unsqueeze(0)  # 第 i 个图像的模型输入尺寸
        )[0][0][0].numpy()

        # 将掩码添加到列表中
        masks.append(mask)
        
    return masks

def sq(w, h):
    return np.concatenate(
        [(np.arange(w * h).reshape(h, w) % w)[:, :, None], (np.arange(w * h).reshape(h, w) // w)[:, :, None]], axis=-1
    )


def mask_to_pos_weighted(mask):
    pos = sq(*image_dims)

    weight = pos[:, :, 0] + pos[:, :, 1]
    weight = weight * weight

    x = np.sum(mask * pos[:, :, 0] * weight) / np.sum(mask * weight)
    y = get_median(mask * weight, 0.95)

    return x, y


def mask_to_pos_naive(mask):
    pos = sq(*image_dims)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())

    return min_pos % image_dims[0] - (image_dims[0] / 16), min_pos // image_dims[0] - (image_dims[0] / 24)


def get_gripper_pos(episode_id, frame, builder, plot=True):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    images = [step["observation"][image_label] for step in episode["steps"]]

    img = Image.fromarray(images[frame].numpy())
    
    predictions = get_bounding_boxes(img)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)

        for prediction in predictions:
            if prediction["score"] < 0.05:
                continue
            box = prediction["box"]
            show_box(box, ax, prediction, "red")

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)

        if plot:
            plt.imshow(mask, alpha=0.5)
            plt.scatter([pos[0]], [pos[1]])
    else:
        print("No valid bounding box")

    if plot:
        plt.show()

COUNTER = 0
def get_gripper_pos_raw(img):
    global COUNTER
    COUNTER += 1
    print("count get_gripper_pos_raw", COUNTER)
    img = Image.fromarray(img.numpy())
    
    predictions = get_bounding_boxes(img)
    
    if len(predictions) > 0:
        
    
        mask = get_gripper_mask(img, predictions[0])
        
        pos = mask_to_pos_naive(mask)
        
    else:
        mask = np.zeros(image_dims)
        pos = (-1, -1)
        predictions = [None]
   
    return (int(pos[0]), int(pos[1])), mask, predictions[0]

def get_results(pos_result, mask_result, predictions, index_nopredictions, nopredict_tuple):
    """
    将 pos_result, mask_result, predictions 列表中相同位置元素组成的元组放入列表 result
    并在 result 的长度等于 index_nopredictions 中的某个元素时，将 nopredict_tuple 放入 result

    参数:
        pos_result (List): 位置结果列表。
        mask_result (List): 掩码结果列表。
        predictions (List): 预测结果列表。
        index_nopredictions (List[int]): 存储多个 index 的列表。
        nopredict_tuple (Tuple): 无预测时的特定元组。

    返回:
        List[Tuple]: 包含所有添加的元素和特定元组的列表。
    """
    # 结果列表
    result = []

    # 遍历 pos_result, mask_result, predictions
    for pos, mask, pred in zip(pos_result, mask_result, predictions):
        # 将相同索引的元素组成元组，并放入列表 result
        result.append((pos, mask, pred))

        # 检查 result 的长度是否等于 index_nopredictions 中的某个元素
        if len(result) in index_nopredictions:
            # 将 nopredict_tuple 放入 result
            result.append(nopredict_tuple)

    return result

def sam_batch_process(images_predict, predictions):
    """
    将 images_predict 和 predictions 列表分成批次，调用 get_gripper_mask 处理，并拼接结果。

    参数:
        images_predict (List): 图像列表。
        predictions (List): 预测结果列表。
        batch_size (int): 每个批次的大小，默认为 16。

    返回:
        List: 所有批次的 masks 拼接结果。
    """
    masks = []

    # 将 images_predict 和 predictions 分成批次
    for i in range(0, len(images_predict), sam_batch_num):
        # 获取当前批次的图像和预测结果
        batch_images = images_predict[i:i + sam_batch_num]
        batch_predictions = predictions[i:i + sam_batch_num]

        
        # 调用 get_gripper_mask 处理当前批次
        batch_masks = get_gripper_mask_batch(batch_images, batch_predictions)
        
        # 将当前批次的 masks 添加到结果列表中
        masks.extend(batch_masks)
        
        print(f'count get gripper mask from image {i} to {i + sam_batch_num}')

    return masks

# owlvit use single img and sam use batch img
def get_gripper_pos_sambatch(images):
    """
    批量处理多个图像，生成机械臂的位置、掩码和检测结果。

    参数:
        images (List[PIL.Image]): 多个图像的列表。

    返回:
        List[Tuple[Tuple[int, int], np.ndarray, dict]]: 每个图像的结果，格式为 ((x, y), mask, prediction)。
    """
    global COUNTER

    
    predictions = []
    images_predict = []
    index_nopredictions = []
    # 遍历每个图像
    for index, img in enumerate(images):
        COUNTER += 1
        print("count get_gripper_pos_raw", COUNTER)

        # 将图像转换为 PIL 格式
        img_pil = Image.fromarray(img.numpy())

        # 获取检测结果
        predict = get_bounding_boxes(img_pil)
        if len(predict) > 0:
           
            predictions.append(predict[0])
            images_predict.append(img_pil)
        else:
            index_nopredictions.append(index)

    masks = sam_batch_process(images_predict, predictions)

    
    mask_result = []
    pos_result = []
    for mask in masks:
        pos = mask_to_pos_naive(mask)
        mask_result.append(mask)
        pos_result.append((int(pos[0]), int(pos[1])))
    
    nopredict_tuple = ((-1, -1), np.zeros(image_dims), None)
    results = get_results(pos_result, mask_result, predictions, index_nopredictions, nopredict_tuple)

    return results

def get_gripper_pos_batch(images):
    """
    批量处理多个图像，生成机械臂的位置、掩码和检测结果。

    参数:
        images (List[PIL.Image]): 多个图像的列表。

    返回:
        List[Tuple[Tuple[int, int], np.ndarray, dict]]: 每个图像的结果，格式为 ((x, y), mask, prediction)。
    """
    global COUNTER

    
    predictions = []
    images_predict = []
    index_nopredictions = []
    
    images = [Image.fromarray(img.numpy()) for img in images]
    
    all_predictions = get_bounding_boxes_batch(images)
    print(f'############################## {len(all_predictions)}')
    for index, prediction in enumerate(all_predictions):
        if len(prediction) > 0:
            predictions.append(prediction[0])
            images_predict.append(images[index])
        else:
            index_nopredictions.append(index)

    masks = sam_batch_process(images_predict, predictions)
    
    mask_result = []
    pos_result = []
    for mask in masks:
        pos = mask_to_pos_naive(mask)
        mask_result.append(mask)
        pos_result.append((int(pos[0]), int(pos[1])))
    
    nopredict_tuple = ((-1, -1), np.zeros(image_dims), None)
    results = get_results(pos_result, mask_result, predictions, index_nopredictions, nopredict_tuple)

    return results

def process_trajectory(episode):
    images = [step["observation"][image_label] for step in episode["steps"]]
    states = [step["observation"]["state"] for step in episode["steps"]]

    results = get_gripper_pos_batch(images)
    raw_trajectory = [(*result, state) for result, state in zip(results, states)]
   
    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))

    prev_found[0] = -1e6
    next_found[-1] = 1e6

    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]

    for i in reversed(range(0, len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]

    if next_found[0] == next_found[-1]:
        # the gripper was never found
        return None

    # Replace the not found positions with the closest neighbor
    for i in range(0, len(raw_trajectory)):
        raw_trajectory[i] = raw_trajectory[prev_found[i] if i - prev_found[i] < next_found[i] - i else next_found[i]]

    return raw_trajectory


def get_corrected_positions(episode_id, builder, plot=False):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    pr_pos = get_corrected_positions_episode(episode, plot=plot)

    return pr_pos


def get_corrected_positions_episode(episode, plot=False):

    t = process_trajectory(episode)
  
    images = [step["observation"][image_label] for step in episode["steps"]]
    images = [img.numpy() for img in images]

    pos = [tr[0] for tr in t]
    
    
    points_2d = np.array(pos, dtype=np.float32)
    points_3d = np.array([tr[-1][:3] for tr in t])

    from sklearn.linear_model import RANSACRegressor

    points_3d_pr = np.concatenate([points_3d, np.ones_like(points_3d[:, :1])], axis=-1)
    points_2d_pr = np.concatenate([points_2d, np.ones_like(points_2d[:, :1])], axis=-1)
    reg = RANSACRegressor(random_state=0).fit(points_3d_pr, points_2d_pr)

    pr_pos = reg.predict(points_3d_pr)[:, :-1].astype(int)

    plot = True
    
    if plot:
        # 创建一个 VideoWriter 对象
        global TIM
        output_file = f"./gripper_positions/output_video_{TIM}.mp4"  # 输出视频文件名
        TIM += 1
        fps = 10  # 帧率
        frame_size = (images[0].shape[1], images[0].shape[0])  # 视频帧的宽度和高度
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        # 在图像上绘制点并写入视频
        for img, p in zip(images, pr_pos):
            img_with_circle = cv2.circle(img, (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
            out.write(img_with_circle)  # 将帧写入视频

        # 释放 VideoWriter 对象
        out.release()
        print(f"视频已保存到 {output_file}")

        
        # 如果需要显示视频
        # mediapy.show_video(images, fps=10)

    return pr_pos

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将NumPy数组转换为列表
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj

if __name__=="__main__":
    # ds = tfds.load("bridge_orig", data_dir="/data/lzx/bridge_dataset", split=f"train[{0}%:{100}%]")
    ds = tfds.load("libero_10_no_noops", data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")
    gripper_positions_json_path = "./gripper_positions/gripper_positions.json"

    start = time.time()
    gripper_positions_json = {}
    
    tim = 0
    
    TIM = 0
    for ep_idx, episode in enumerate(ds):

        episode_id = ep_idx
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")

        # try:
        pr_pos = get_corrected_positions_episode(episode)
    
        if file_path not in gripper_positions_json.keys():
            gripper_positions_json[file_path] = {}

        gripper_positions_json[file_path][int(episode_id)] = pr_pos
        end = time.time()
    
        # 转换数据
        gripper_positions_json = convert_ndarray_to_list(gripper_positions_json)
            
        print(f"finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")

        if tim == 10:
            break
        tim += 1
        # except:
        #     print('!!!!!!!!!!!!!!!!!!!!!')
    with open(gripper_positions_json_path, "w") as f:
            json.dump(gripper_positions_json, f, cls=NumpyFloatValuesEncoder)