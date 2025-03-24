export CUDA_VISIBLE_DEVICES=0
export GOOGLE_API_KEY=AIzaSyAl6_EVlpP40-0NYQeeLOU8ggADf3xO4Go

python libero_scripts/label_pipeline.py \
--libero_task_suite libero_10_no_noops \
--libero_dataset_dir /home/nus/LabelCoT/LIBERO/libero/datasets/libero_10_w_mask \
--action_horizon 10 \
--vlm_model_path prism-dinosiglip+7b \
--batch_size 1 \
--api_provider gemini \
--enable_gripper_positions \
--enable_plan_subtasks \
--enable_bboxes \
--enable_scene_desc \
--enable_primitives \
--enable_merge \
--results_dir /data/lx/libero_dataset/openvla_modified/cot \
