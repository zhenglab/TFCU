import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def create_base_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    # for i in range(1, num_segments):
    #     additional_targets[f'image{i}'] = 'image'
    if split == 'train':
        base_transform = alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            alb.HorizontalFlip(),
            alb.augmentations.transforms.ToGray(p=0.01),
            alb.Resize(args.image_size, args.image_size),
            # alb.RandomResizedCrop(args.image_size, args.image_size,scale=(0.2, 1), p=1),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'test':
        # if args.type == 'RGBShift':
        #     level = args.level*50
        #     base_transform = alb.Compose([
        #         alb.RGBShift((-level,level),(-level,level),(-level,level),p=1.0),  # Level 1-5 依次：
        #         alb.Resize(args.image_size, args.image_size),
        #         alb.Normalize(mean=args.mean, std=args.std),
        #         ToTensorV2(),
        #     ], additional_targets=additional_targets)
        # elif args.type == 'ImageCompression':
        #     level = args.level*10
        #     base_transform = alb.Compose([
        #         alb.ImageCompression(quality_lower=60-level,quality_upper=100-level,p=1.0),
        #         alb.Resize(args.image_size, args.image_size),
        #         alb.Normalize(mean=args.mean, std=args.std),
        #         ToTensorV2(),
        #     ], additional_targets=additional_targets)
        # elif args.type == 'brightness':
        #     level = args.level*0.2
        #     base_transform = alb.Compose([
        #         alb.RandomBrightnessContrast(brightness_limit=(-level,level), contrast_limit=(-level,level), p=1.0),
        #         alb.Resize(args.image_size, args.image_size),
        #         alb.Normalize(mean=args.mean, std=args.std),
        #         ToTensorV2(),
        #     ], additional_targets=additional_targets)
        # elif args.type == 'ToGray':
        #     level = args.level*0.2
        #     base_transform = alb.Compose([
        #         alb.augmentations.transforms.ToGray(p=level),
        #         alb.Resize(args.image_size, args.image_size),
        #         alb.Normalize(mean=args.mean, std=args.std),
        #         ToTensorV2(),
        #     ], additional_targets=additional_targets)            

        # level = 1.0    
        base_transform = alb.Compose([
            # alb.augmentations.transforms.ToGray(p=1.0),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)


    return base_transform


def create_all_transforms(args, split='train'):
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    if split == 'train':
        base_transform = alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            alb.HorizontalFlip(),
            alb.augmentations.transforms.ToGray(p=0.01),
            alb.Resize(args.image_size, args.image_size),
        ], additional_targets=additional_targets)



def create_stu_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    if split == 'train':
        base_transform = alb.Compose([
            # alb.RandomResizedCrop(args.image_size, args.image_size,scale=(0.4, 1), p=1),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    return base_transform


def create_totensor_transforms(args, split='train'):
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    base_transform = alb.Compose([
        alb.Normalize(mean=args.mean, std=args.std),
        ToTensorV2(),
    ], additional_targets=additional_targets)
    return base_transform


def create_base_transforms_strong(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    if split == 'train':
        base_transform = alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.5),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.5),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.7),
            alb.HorizontalFlip(),
            alb.augmentations.transforms.ToGray(p=0.02),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    return base_transform
def create_base_transformsv2(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    # for i in range(1, num_segments):
    #     additional_targets[f'image{i}'] = 'image'
    if split == 'train':
        base_transform = alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.2),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.2),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.1),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.1),
            alb.HorizontalFlip(p=0.5),
            alb.augmentations.transforms.ToGray(p=0.01),
            alb.Resize(args.image_size, args.image_size),
            # alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return base_transform


def create_base_sbi_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    # num_segments = args.num_segments if 'num_segments' in args else 1
    # additional_targets = {}
    # for i in range(1, num_segments):
    #     additional_targets[f'image{i}'] = 'image'

    if split == 'train':
        base_transform = alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
			ToTensorV2(),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

    return base_transform

# def create_train_transforms(size=300):
#     return Compose([
#         ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
#         GaussNoise(p=0.1),
#         GaussianBlur(blur_limit=3, p=0.05),
#         HorizontalFlip(),
#         OneOf([
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
#         ], p=1),
#         PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
#         OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
#         ToGray(p=0.2),
#         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
#     ]
#     )
