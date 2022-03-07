from PIL import Image
from torch import from_numpy, load, no_grad
from torch.autograd import Variable
from torch.backends.cudnn import benchmark as cudnn_benchmark
from torch.cuda import empty_cache as empty_cuda_cache
from torch.nn import DataParallel
import torchvision
import torch


def pre_process(img:torch.Tensor, size:tuple, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))->torch.Tensor:
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Resize(size),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])(img)


def inv_normalization(img:torch.Tensor)->Image:
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1/0.229, 1/0.224, 1/0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1., 1., 1.]),
        torchvision.transforms.ToPILImage()
    ])(img)