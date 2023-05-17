export TORCH_DIST_INIT_BARRIER=0
python3 main.py \
	container_mounts=["/home/ubuntu/NeMo-Megatron-Launcher/NeMo:/opt/NeMo"]
