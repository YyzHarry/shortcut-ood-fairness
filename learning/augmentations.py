from torchvision import transforms


def get_image_aug(aug_type, target_shape, scale_size):
    if aug_type == "basic":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == 'basic2':
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "auto_aug":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "rand_aug":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "trivial_aug":
        transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(target_shape[0]),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "augmix":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError(f"Augmentation type [{aug_type}] not supported.")
    return transform
