### Model Settings
pretrained_model_name_or_path = "/home/jovyan/boomcheng-data-shcdt/chengbo/MODEL_DiT/FLUX.1-dev"
revision = None
variant = None
cache_dir = None

### Training Settings
seed = 42
report_to = "wandb"
tracker_project_name = "multilayer"
wandb_job_name = "YOU_FORGET_TO_SET"
logging_dir = "logs"
max_train_steps = None
checkpoints_total_limit = None

# gpu
allow_tf32 = True
gradient_checkpointing = True
mixed_precision = "bf16"
