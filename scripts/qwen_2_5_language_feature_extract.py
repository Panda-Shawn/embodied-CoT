from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import h5py
from tqdm import tqdm
import tensorflow_datasets as tfds


def extract_qwen25_language_embedding(input_language, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": input_language}
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True
    )
    
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True).to(model.device)
    offsets = encoding.offset_mapping[0]

    start_marker = "<|im_start|>user"
    end_marker = "<|im_end|>"
    start_pos = full_text.find(start_marker)
    end_pos = full_text.find(end_marker, start_pos)
    if start_pos == -1 or end_pos == -1:
        raise ValueError("无法在文本中找到用户输入标记")
    
    user_text_start = start_pos + len(start_marker)

    token_start, token_end = None, None
    for i, (s, e) in enumerate(offsets.tolist()):
        if token_start is None and s >= user_text_start:
            token_start = i
        if token_start is not None and e >= end_pos:
            token_end = i + 1
            break
    
    if token_start is None or token_end is None:
        raise ValueError("未能定位到用户输入对应的 token 范围")

    outputs = model(**{k: v for k, v in encoding.items() if k != "offset_mapping"}, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    user_embedding = last_hidden_state[:, token_start:token_end, :]
    return user_embedding.cpu().detach()

# def clean_path(path):
#     return path.replace("/", "_").replace("\\", "_")

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype="auto",
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reasoning_path = "/home/nus/Libero/cot_language_v1/cot_language_v1/libero_10/reasoning_224.json"
    reasoning = json.load(open(reasoning_path, "r"))

    # check and find that the task description is the same as the task in reasoning.
    # ds = tfds.load('libero_10', data_dir="/home/nus/Libero/end_to_end", split='train')
    # for ep_idx, episode in enumerate(ds):
    #     episode_id = episode["episode_metadata"]["episode_id"].numpy().decode()
    #     file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    #     print(file_path)
    #     for step_idx, step in enumerate(episode["steps"]):
    #         lang_instruction = step["language_instruction"].numpy().decode()
    #         print(lang_instruction)
    #         break
    #     break
    # exit()

    hdf5_path = "qwen_2_5_language_embedding.h5"
    h5f = h5py.File(hdf5_path, "w")

    num_moves = 16
    pbar = tqdm(total=len(reasoning))
    pbar.set_description("Extracting Qwen2.5 embeddings")

    for i, file_path in enumerate(reasoning):
        # if i == 2:
        #     break

        len_episode = len(reasoning[file_path]["0"]["reasoning"].keys())
        # file_group_name = clean_path(file_path)
        episode_group = h5f.create_group(file_path)
        reasoning_group = episode_group.create_group("0")

        task_description = reasoning[file_path]["0"]["reasoning"]["0"]["task"]
        task_embedding = extract_qwen25_language_embedding(task_description, model, tokenizer)
        # shape: (1, token_len, 3584)
        task_embedding = task_embedding.to(torch.float32)
        reasoning_group.create_dataset(
            "language_description",
            data=task_embedding.numpy(),
            compression="gzip"
        )
        planning_group = reasoning_group.create_group("language_planning")

        for j in range(len_episode):
            task = reasoning[file_path]["0"]["reasoning"][str(j)]["task"]
            plan = reasoning[file_path]["0"]["reasoning"][str(j)]["plan"]
            subtask = reasoning[file_path]["0"]["reasoning"][str(j)]["subtask"]
            subtask_reason = reasoning[file_path]["0"]["reasoning"][str(j)]["subtask_reason"]
            move = reasoning[file_path]["0"]["reasoning"][str(j)]["move"]
            move_reason = reasoning[file_path]["0"]["reasoning"][str(j)]["move_reason"]

            bboxes = reasoning[file_path]["0"]["features"]["bboxes"][j]
            visible_objects = ", ".join(f"{name} {str(bbox)}" for name, bbox in bboxes)

            gripper_positions = reasoning[file_path]["0"]["features"]["gripper_position"]
            future_positions = []
            for k in range(num_moves):
                if j + k < len(gripper_positions):
                    future_positions += gripper_positions[j + k]
                else:
                    future_positions += future_positions[-2:] if future_positions else [0.0, 0.0]

            input_language = f"TASK: {task} PLAN: {plan} VISIBLE_OBJECTS: {visible_objects} SUBTASK REASONING: {subtask_reason} SUBTASK: {subtask} MOVE REASONING: {move_reason} MOVE: {move} GRIPPER_POSITIONS: {future_positions} GRIPPER_POSITIONS: {future_positions}"

            language_embedding = extract_qwen25_language_embedding(input_language, model, tokenizer)
            # shape: (1, token_len, 3584)
            language_embedding = language_embedding.to(torch.float32)

            step_name = str(j)
            planning_group.create_dataset(
                step_name,
                data=language_embedding.numpy(),
                compression="gzip"
            )

        pbar.update(1)

    h5f.close()
    pbar.close()
    print(f"Embeddings saved in HDF5 format to {hdf5_path}")
