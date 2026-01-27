from PIL import Image
import torch
from torch.utils.data import Dataset # 数据集基类，自定义数据集需继承此类

# 接收图像文件路径列表和对应的类别列表
# 支持对图像进行自定义的变换（如 resize、归一化等）
# 实现了数据集的基本功能，如获取数据集长度、根据索引获取单个样本（图像及其类别）
# 自定义了批次数据的整理方式（collate_fn），将多个样本打包成一个批次

# 该类继承自 torch.utils.data.Dataset，必须重写 __init__、__len__、__getitem__ 三个方法，还自定义了 collate_fn 静态方法。
# 核心作用是将原始图像文件和其类别标签组织成 PyTorch 可以直接使用的数据格式。
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    # 根据索引 item 获取单个样本（图像 + 类别），是数据集的核心方法。
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片 # 检查图像是否为RGB模式，若不是则抛出异常（确保数据格式统一）
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # 获取当前图像对应的类别
        label = self.images_class[item]

        # 若指定了图像变换，则对图像执行变换（如 resize、转Tensor、归一化等）
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    # 静态方法，自定义批次数据的整理逻辑。
    # 当 DataLoader 从数据集获取多个样本（组成 batch）时，会调用该函数将这些样本打包成统一的批次格式。
    @staticmethod 
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # batch是一个列表，每个元素是__getitem__返回的元组（img, label）
        # 将batch中的图像和类别分别打包成元组
        images, labels = tuple(zip(*batch))

        # 将多个图像Tensor堆叠成一个批次Tensor（维度：[batch_size, C, H, W]）
        images = torch.stack(images, dim=0)
        # 将类别元组转换为Tensor（维度：[batch_size]）
        labels = torch.as_tensor(labels)
        return images, labels
