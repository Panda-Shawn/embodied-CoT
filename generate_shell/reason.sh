export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

python colosseum_scripts/batch_generate_plan_subtasks.py \
--batch_size 1 \
--action_horizon 10 \
--scene_desc_path /home/nus/Colosseum/cot/scene_descriptions.json \
--primitives_path /home/nus/Colosseum/cot/primitives_h10.json \
--dataset_dir /home/nus/Colosseum \
# --force_regenerate