# Training settings
batch_size = 32
epochs = 120
lr = 0.001
opt = 'Adam'
deepcoral = True
momentum = 0.9
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./dataset/"
source_name = "office31_source"
target_name = "office31_target"
use_checkpoint = False
image_size = [128, 128]
weighted = True
