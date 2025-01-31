import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return {'image': (img1, img2),
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img1 = img1.rotate(rotate_degree, Image.BILINEAR)
        img2 = img2.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}



class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2),
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}
    
import random
import numpy as np
import torch
from PIL import Image


class CutMixBoxAugmentation(object):
    def __init__(self, img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
        self.img_size = img_size
        self.p = p
        self.size_min = size_min
        self.size_max = size_max
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        cutmix_mask = obtain_cutmix_box(self.img_size, self.p, self.size_min, self.size_max, self.ratio_1, self.ratio_2)

        # 将 cutmix_mask 转换为 PIL 图像以便与 img1 和 img2 进行操作
        cutmix_mask_pil = Image.fromarray((cutmix_mask.numpy() * 255).astype(np.uint8))
        cutmix_mask_pil = cutmix_mask_pil.resize(img1.size, Image.NEAREST)
        cutmix_mask_tensor = torch.from_numpy(np.array(cutmix_mask_pil) / 255).unsqueeze(0).float()

        # 进行 CutMix 操作
        img1 = img1 * (1 - cutmix_mask_tensor) + img2 * cutmix_mask_tensor

        return {'image': (img1.squeeze(0),),
                'label': mask}


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


# # 示例用法
# if __name__ == "__main__":
#     # 假设 sample 是一个包含 'image' 和 'label' 的字典
#     sample = {'image': (Image.new('RGB', (200, 200)), Image.new('RGB', (200, 200))),
#               'label': Image.new('L', (200, 200))}
#     cutmix_aug = CutMixBoxAugmentation(img_size=200)
#     new_sample = cutmix_aug(sample)

import random
from PIL import Image
from torchvision import transforms


class ColorJitterAugmentation(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        img1 = self.color_jitter(img1)
        img2 = self.color_jitter(img2)

        return {'image': (img1, img2),
                'label': mask}


train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

test_transforms = transforms.Compose([
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomFixRotate(),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

# weak_argument= transforms.Compose([
#             RandomHorizontalFlip(),
#             RandomVerticalFlip(),
#             # RandomFixRotate(),
#             # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
#             # RandomGaussianBlur(),
#             # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ])


# strong_argument= transforms.Compose([
#             # RandomHorizontalFlip(),
#             ColorJitterAugmentation(),
#             # RandomVerticalFlip(),
#             # RandomFixRotate(),
#             # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
#             RandomGaussianBlur(),
#             # CutMixBoxAugmentation(256),
#             # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ])


import torch
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip


# def weak_argument(img):
#     """
#     对输入的图像张量进行水平翻转和垂直翻转的弱增强操作。

#     参数:
#     img (torch.Tensor): 输入的图像张量，维度通常为[batch_size, channels, height, width]。

#     返回:
#     torch.Tensor: 经过水平翻转和垂直翻转增强后的图像张量，维度与输入保持一致。
#     """
#     # 水平翻转
#     horizontal_flip_transform = RandomHorizontalFlip(p=0.5)  # p=0.5表示有50%的概率进行水平翻转
#     img = horizontal_flip_transform(img)

#     # 垂直翻转
#     vertical_flip_transform = RandomVerticalFlip(p=0.5)  # p=0.5表示有50%的概率进行垂直翻转
#     img = vertical_flip_transform(img)

#     return img






# # 示例用法
# if __name__ == "__main__":
#     # 假设 sample 是一个包含 'image' 和 'label' 的字典
#     sample = {'image': (Image.new('RGB', (200, 200)), Image.new('RGB', (200, 200))),
#               'label': Image.new('L', (200, 200))}
#     color_jitter_aug = ColorJitterAugmentation()
#     new_sample = color_jitter_aug(sample)


'''
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
'''
