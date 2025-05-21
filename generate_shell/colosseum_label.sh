export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

task=${1:-10}
python colosseum_scripts/batch_generate_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--dataset_dir /data2/lzixuan/colosseum_dir \
--primitives_path /data2/lzixuan/colosseum_dir/cot/primitives_h10.json \
--scene_desc_path /data2/lzixuan/colosseum_dir/cot/scene_descriptions.json \
--force_regenerate