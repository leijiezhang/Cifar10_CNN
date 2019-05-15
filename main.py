from train_main import _train

# some setting about the net
epoch_num = 40
lr = 0.05
batch_size = 20
backbone = 'resnetn_101' # alexnet vgg16 resnet18 resnet50 resnetn_101 inception_v3  densenet121
checkpoint_path = './models'
results_path = './results'
data_dir = './data'
data_name = 'cifar10'
use_checkpoint = False
checkpoints_name = 'vgg16-0.9010-0.0100.pth'
is_pre_train = False
is_inception = False
opt_type = 'SGD'
is_lr_adjust = True
lr_adjust_mtd = 'Cos'

cuda_name = 'cuda:0'

_train(epoch_num, lr, batch_size, backbone, data_dir, data_name,
       use_checkpoint, checkpoint_path, checkpoints_name, is_inception,
       is_pre_train, results_path, cuda_name,
       opt_type, is_lr_adjust, lr_adjust_mtd)
