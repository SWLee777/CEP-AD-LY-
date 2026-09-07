import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def compute_gradient_foreground(image, patch_size=14):

    gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # [B, 1, H, W]

    patch_grad = F.avg_pool2d(gradient_mag, kernel_size=patch_size, stride=patch_size)  # [B, 1, h, w]
    patch_grad = patch_grad.squeeze(1)  # [B, h, w]

    B = patch_grad.shape[0]
    patch_grad = patch_grad.reshape(B, -1)  # [B, num_patches]

    min_val = patch_grad.min(dim=1, keepdim=True).values
    max_val = patch_grad.max(dim=1, keepdim=True).values
    patch_grad = (patch_grad - min_val) / (max_val - min_val + 1e-8)

    return patch_grad.T  # [num_patches, B]


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform
