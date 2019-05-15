import os
import torch
from load_data import _load_data
from load_net import _load_net
import torch.nn as nn
import torch.optim as optim
from train_model import _train_model
from get_transformer import _get_transformer



def _train(epoch_num: int, lr: float, batch_size: int, backbone: str, data_dir: str,data_name: str,
           use_checkpoint: bool, checkpoint_path: str, checkpoints_name: str,
           is_inception: bool,  is_pre_train: bool, results_path: str, cuda_name: str,
           opt_type: str, is_lr_adjust: bool, lr_adjust_mtd: str):
    # some setting about the net
    epoch_num = epoch_num
    lr = lr
    batch_size = batch_size
    backbone = backbone  # alexnet vgg11 inception_v3 resnet18 densenet121

    is_pre_train = is_pre_train

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    if torch.cuda.is_available:
        device = torch.device(cuda_name)
    else:
        device = torch.device("cpu")
    print(device)

    # get transformer
    transform_test, transform_train = _get_transformer()
    # get data
    trainloader, testloader, classes, _, _ = _load_data(data_dir, data_name, transform_train, transform_test, batch_size)
    n_class = len(classes)
    # get net model
    net = _load_net(use_checkpoint, checkpoint_path, checkpoints_name, n_class, device, backbone, is_pre_train)
    criterion = nn.CrossEntropyLoss()

    dataloaders = {'train': trainloader, 'val': testloader}

    # train model
    net, best_acc, best_epoch, output_arr_val, output_arr_train, f_imgs_val = _train_model(lr, device, net, dataloaders,
                                                                           criterion, opt_type, lr_adjust_mtd, is_lr_adjust, epoch_num, is_inception)
    checkpoints_dir = os.path.join(checkpoint_path,
                                       '{:s}-{:.4f}-{:.4f}.pth'.format(backbone, best_acc, lr))
    torch.save(net.state_dict(), checkpoints_dir)
    print(f'Model has been saved to {checkpoints_dir}')

    # save the result of this training
    results_dic = {'best_acc': best_acc,
                   'best_epoch': best_epoch,
                   'output_arr_val': output_arr_val,
                   'error_img_msg': f_imgs_val,
                   'output_arr_train': output_arr_train}
    results_dir = os.path.join(results_path,
                                       '{:s}-{:.4f}-{:.4f}.pt'.format(backbone, best_acc, lr))
    torch.save(results_dic, results_dir)
