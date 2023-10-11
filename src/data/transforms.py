import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def blur_transforms(p=0.5, blur_limit=5):
    """
    Applies MotionBlur or GaussianBlur random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
        blur_limit (int, optional): Blur intensity limit. Defaults to 5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(always_apply=True),
            albu.GaussianBlur(always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(50, 150), always_apply=True),
            albu.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.2, always_apply=True
            ),
        ],
        p=p,
    )


def distortion_transforms(p=0.5):
    """
    Applies ElasticTransform with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=1,
                sigma=5,
                alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        ],
        p=p,
    )


def get_transfos(augment=True, resize=None, crop=False, mean=0, std=1, strength=1):
    """
    Returns transformations. todo

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        mean (np array, optional): Mean for normalization. Defaults to 0.
        std (np array, optional): Standard deviation for normalization. Defaults to 1.

    Returns:
        albumentation transforms: transforms.
    """
    if resize is None:
        resize_aug = []
    elif not crop:
        resize_aug = [albu.Resize(resize[0], resize[1])]
    else:
        if resize[0] == 384:
            resize_aug = [albu.Compose([
                albu.LongestMaxSize(512),
                albu.PadIfNeeded(resize[0], resize[1], border_mode=0),
                albu.CenterCrop(resize[0], resize[1]),
            ])]
        else:
            resize_aug = [albu.Compose([
                albu.LongestMaxSize(512),
                albu.PadIfNeeded(384, 384, border_mode=0),
                albu.CenterCrop(384, 384),
                albu.Resize(resize[0], resize[1])
            ])]

    normalizer = albu.Compose(
        resize_aug
        + [
            #             albu.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        p=1,
    )

    if augment:
        if strength == 0:
            augs = [
                albu.HorizontalFlip(p=0.5),
            ]
        elif strength == 1:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.1,
                    shift_limit=0.0,
                    rotate_limit=20,
                    p=0.5,
                ),
                color_transforms(p=0.5),
            ]
        elif strength == 2:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.1,
                    shift_limit=0.0,
                    rotate_limit=20,
                    p=0.5,
                ),
                color_transforms(p=0.5),
            ]
        elif strength == 3:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.0,
                    rotate_limit=30,
                    p=0.75,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
                distortion_transforms(p=0.25),
            ]
        elif strength == 4:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.,
                    shift_limit=0.,
                    rotate_limit=45,
                    p=0.75,
                ),
                color_transforms(p=0.25),
                blur_transforms(p=0.25),
                distortion_transforms(p=0.5),
            ]
        elif strength == 5:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.1,
                    rotate_limit=45,
                    p=0.75,
                ),
                color_transforms(p=0.25),
                blur_transforms(p=0.25),
                distortion_transforms(p=0.5),
            ]
    else:
        augs = []

    return albu.Compose(augs + [normalizer])
