import torchvision
import torch
import os


def _load_net(use_checkpoint, checkpoint_path, checkpoints_name, n_class, device, backbone, is_pre_train):
    Nets = {
        'alexnet': torchvision.models.alexnet(num_classes=n_class, pretrained=is_pre_train),
        'vgg16': torchvision.models.vgg16(num_classes=n_class, pretrained=is_pre_train),
        'resnet18': torchvision.models.resnet18(num_classes=n_class, pretrained=is_pre_train),
        'resnet50': torchvision.models.resnet50(num_classes=n_class, pretrained=is_pre_train),
        'resnetn_101': torchvision.models.resnet101(num_classes=n_class, pretrained=is_pre_train),
        'densenet121': torchvision.models.densenet121(num_classes=n_class, pretrained=is_pre_train),
        'inception_v3': torchvision.models.inception_v3(num_classes=n_class, pretrained=is_pre_train)
    }
    if (use_checkpoint == True):
        checkpoints_dir = os.path.join(checkpoint_path,
                                           '{:s}'.format(checkpoints_name))
        net = Nets[backbone]
        net.load_state_dict(torch.load(checkpoints_dir, map_location=device))
        # net.eval()
        net.to(device)
    else:
        net = Nets[backbone]
        net.to(device)

    return net