export CUDA_VISIBLE_DEVICES=0

python libero_scripts/regenerate_libero_dataset.py \
--libero_task_suite libero_object \
--libero_raw_data_dir /data/zzb/libero_dataset/raw/libero_object \
--libero_target_dir /data/lx/libero_dataset/openvla_modified/libero_object_mask_object \
--metadata_json_output_path /data/lx/libero_dataset/openvla_modified/libero_object_metadata_mask_object.json \
--resolution 256

# python libero_scripts/regenerate_libero_dataset.py \
# --libero_task_suite libero_spatial \
# --libero_raw_data_dir /data/zzb/libero_dataset/raw/libero_spatial \
# --libero_target_dir /data/lx/libero_dataset/openvla_modified/libero_spatial \
# --metadata_json_output_path /data/lx/libero_dataset/openvla_modified/libero_spatial_metadata.json \
# --resolution 224
