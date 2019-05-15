import torchvision.transforms as transforms


def _get_transformer():
    # init the transform
    transform_test = transforms.Compose(
        [transforms.Resize(252),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train = transforms.Compose(
        [  # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(int(32 * 1.2)),
            transforms.RandomAffine(10, (0, 0.1)),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(),
            # transforms.TenCrop(32),
            transforms.Resize(252),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_test, transform_train
