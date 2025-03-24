export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

task=${1:-10}
python colosseum_scripts/batch_filter_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--dataset_dir /home/nus/Colosseum \
--primitives_path /home/nus/Colosseum/cot/primitives_h10.json \
--scene_desc_path /home/nus/Colosseum/cot/scene_descriptions.json \
--plan_subtasks_path /home/nus/Colosseum/cot/chain_of_thought_h10.json \
--force_regenerate