import torchvision
import torch


def _load_data(data_dir, data_name, transform_train, transform_test, batch_size):
    # download datasset
    if(data_name == 'cifar10'):
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        print('other datasets')

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, trainset, testset
