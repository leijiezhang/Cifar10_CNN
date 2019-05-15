import argparse
from train_main import _train


# python3 main_console.py -e 20 -s 60 -l 0.0005 -b 'vgg16' -c 3
backbone_option = {'alexnet', 'vgg16', 'resnet18', 'resnet50', 'resnetn_101', 'inception_v3', 'densenet121'}
cuda_option = {0, 1, 2, 3}

# some setting about the net
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch_num', type=int,  required=True, help='number of epoch')
parser.add_argument('-s', '--batch_size', type=int,  required=True, help='size of batch')
parser.add_argument('-l', '--learning_rate', type=float,  required=True, help='value of learning rate')
parser.add_argument('-b', '--backbone', type=str, choices=backbone_option, required=True, help='name of backbone model')
parser.add_argument('-c', '--cuda_num', type=int, choices=cuda_option, required=True, help='number of gpu device')
parser.add_argument('-o', '--path_to_result_dir', type=str, default='./results', help='path to outputs directory')
parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
parser.add_argument('-a', '--data_name', type=str, default='cifar10', help='data file name')
parser.add_argument('-r', '--path_to_checkpoint_dir', type=str, default='./models', help='path to resuming checkpoint')
parser.add_argument('-f', '--check_points_file', type=str, default='', help='the checkpoint file')
parser.add_argument('-v', '--is_direct_val', type=bool, default=False, help='whether to validate the model directly')
parser.add_argument('-p', '--is_pre_train', type=bool, default=False, help='whether to pre training')
parser.add_argument('-i', '--is_inception', type=bool, default=False, help='whether the backbone is inception net ')
parser.add_argument('-u', '--use_checkpoint', type=bool, default=False, help='whether to load check point ')
parser.add_argument('-n', '--is_lr_adjust', type=bool, default=False, help='whether to adjust the learning rate auto ')
parser.add_argument('-m', '--lr_adjust_mtd', type=str, default='Cos', help='learning rate adjustment method')
parser.add_argument('-t', '--opt_type', type=str, default='SGD', help='optimization type')
args = parser.parse_args()

# some setting about the net
epoch_num = args.epoch_num
lr = args.learning_rate
batch_size = args.batch_size
backbone = args.backbone # alexnet vgg16 resnet18 resnet50 resnetn_101 inception_v3  densenet121
checkpoint_path = args.path_to_checkpoint_dir
results_path = args.path_to_result_dir
data_dir = args.data_dir
data_name = args.data_dir
use_checkpoint = args.use_checkpoint
checkpoints_name = args.check_points_file
is_pre_train = args.is_pre_train
is_inception = args.is_inception
opt_type = args.opt_type
is_lr_adjust = args.is_lr_adjust
lr_adjust_mtd = args.lr_adjust_mtd
cuda_name = f'cuda:{args.cuda_num}'

_train(epoch_num, lr, batch_size, backbone, data_dir, data_name,
       use_checkpoint, checkpoint_path, checkpoints_name, is_inception,
       is_pre_train, results_path, cuda_name,
       opt_type, is_lr_adjust, lr_adjust_mtd)
