export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

task=${1:-10}
python furniture_bench_scripts/batch_filter_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--dataset_dir /data2/lzixuan/furniture-bench/scripted_sim_demo/${task} \
--primitives_path /data2/lzixuan/furniture-bench/scripted_sim_demo/${task}/cot/${task}_primitives.json \
--scene_desc_path /data2/lzixuan/furniture-bench/scripted_sim_demo/${task}/cot/${task}_scene_desc.json \
--plan_subtasks_path /data2/lzixuan/furniture-bench/scripted_sim_demo/${task}/cot/${task}_plan_subtasks.json \
--force_regenerate