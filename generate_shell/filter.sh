export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

python libero_scripts/batch_filter_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--libero_dataset_dir /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_10_w_mask \
--libero_scene_desc_path /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_10_w_mask/cot/libero_10_scene_desc.json \
--libero_plan_subtasks_path /data2/lzixuan/embodied-CoT/libero_scripts/final_reasonings/libero_10_w_mask/cot/libero_10_plan_subtasks.json \
--force_regenerate