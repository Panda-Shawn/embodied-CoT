export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

python furniture_bench_scripts/label_pipeline.py \
--furniture one_leg \
--dataset_dir /data2/lzixuan/furniture-bench/scripted_sim_demo/one_leg \
--action_horizon 10 \
--vlm_model_path /data2/lzixuan/.cache/huggingface/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/prism-dinosiglip+7b \
--batch_size 1 \
--api_provider gemini \
--enable_merge \
# --enable_plan_subtasks \
# --enable_gripper_positions \
# --enable_bboxes \
# --enable_primitives \
# --enable_scene_desc \
