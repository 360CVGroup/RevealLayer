_base_ = "./base.py"

### path & device settings

output_path_base = "./output/"
cache_dir = None


### wandb settings
wandb_job_name = "flux_" + '{{fileBasenameNoExtension}}'

resolution = 1024

### Model Settings
rank = 64
text_encoder_rank = 64
train_text_encoder = False
max_layer_num = 12
learnable_proj = True

### Training Settings
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
guidance_scale = 1.0 ###IMPORTANT
layer_weighting = 5.0

# steps
train_batch_size = 1
num_train_epochs = 1
max_train_steps = None
checkpointing_steps = 2000
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 1

# lr
optimizer = "prodigy"
learning_rate = 1.0
scale_lr = False
lr_scheduler = "constant"
lr_warmup_steps = 0
lr_num_cycles = 1
lr_power = 1.0

# optim
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-3
adam_epsilon = 1e-8
prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True
max_grad_norm = 1.0

# logging
tracker_task_name = '{{fileBasenameNoExtension}}'
output_dir = output_path_base + "{{fileBasenameNoExtension}}"

### Validation Settings
num_validation_images = 1
validation_steps = 2000