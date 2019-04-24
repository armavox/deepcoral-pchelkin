# Training settings
epochs = 128

lr = 0.001
batch_size = 8
momentum = 0.9
l2_decay = 5e-4

opt = 'Adam'
deepcoral = True

seed = 8
log_interval = 100  # Log every N batches

train_path = "./dataset/"
val_path = "./dataset/"
source_name = "office31_source"
target_name = "office31_target"
use_checkpoint = False
image_size = [64, 64]
weighted = False

small_dataset = False
small_size = 0.02
