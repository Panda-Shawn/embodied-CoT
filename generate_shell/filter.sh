export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

task=${1:-10}
python libero_scripts/batch_filter_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--libero_dataset_dir /data2/lzixuan/LabelCoT/LIBERO/libero/datasets/libero_90_no_noops \
--libero_primitives_path /data2/lzixuan/LabelCoT/LIBERO/libero/datasets/libero_90_no_noops/cot/libero_90_no_noops_primitives.json \
--libero_scene_desc_path /data2/lzixuan/LabelCoT/LIBERO/libero/datasets/libero_90_no_noops/cot/libero_90_no_noops_scene_desc.json \
--libero_plan_subtasks_path /data2/lzixuan/LabelCoT/LIBERO/libero/datasets/libero_90_no_noops/cot/libero_90_no_noops_plan_subtasks.json \
--force_regenerate