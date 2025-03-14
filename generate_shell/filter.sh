export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

task=${1:-10}
python libero_scripts/batch_filter_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--libero_dataset_dir /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_${task}_w_mask \
--libero_primitives_path /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_${task}_w_mask/cot/libero_${task}_primitives.json \
--libero_scene_desc_path /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_${task}_w_mask/cot/libero_${task}_scene_desc.json \
--libero_plan_subtasks_path /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_${task}_w_mask/cot/libero_${task}_plan_subtasks.json \
--force_regenerate