# Training settings
epochs = 1

lr = 0.01
batch_size = 8
momentum = 0.9
l2_decay = 5e-4

opt = 'SGD'
deepcoral = False

seed = 8
log_interval = 100  # Log every N batches

train_path = "./dataset/"
val_path = "./dataset/"
source_name = "luna_nodules"
target_name = "source"
use_checkpoint = False
image_size = [64, 64]
weighted = False

small_dataset = True
small_size = 0.1
