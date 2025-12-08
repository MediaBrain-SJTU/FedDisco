import os
import os.path
import logging
import numpy as np
import paddle.io as data
from PIL import Image
from paddle.vision.datasets import Cifar10, Cifar100, FashionMNIST
from paddle.io import Dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def default_loader(path):
    """默认的图像加载函数"""
    return Image.open(path).convert('RGB')


def find_classes(dir):
    """查找目录中的所有类别"""
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    """创建数据集列表"""
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if any(fname.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class ImageFolder(Dataset):
    """PaddlePaddle版本的ImageFolder数据集"""

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root + "\n")

        self.root = root
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class CIFAR10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # PaddlePaddle的CIFAR10数据集
        # 注意：根据PaddlePaddle版本，可能需要调整数据加载方式
        # PaddlePaddle的CIFAR10使用mode参数：'train'或'test'
        cifar_dataobj = Cifar10(self.root, mode='train' if self.train else 'test', transform=None)

        # 尝试不同的方式获取数据和标签
        if hasattr(cifar_dataobj, 'data') and hasattr(cifar_dataobj, 'targets'):
            data = np.array(cifar_dataobj.data)
            target = np.array(cifar_dataobj.targets)
        elif hasattr(cifar_dataobj, 'images') and hasattr(cifar_dataobj, 'labels'):
            data = np.array(cifar_dataobj.images)
            target = np.array(cifar_dataobj.labels)
        else:
            # 如果以上属性都不存在，需要通过迭代获取
            data = []
            target = []
            for i in range(len(cifar_dataobj)):
                img, label = cifar_dataobj[i]
                data.append(np.array(img))
                target.append(label)
            data = np.array(data)
            target = np.array(target)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # PaddlePaddle的CIFAR100数据集
        cifar_dataobj = Cifar100(self.root, mode='train' if self.train else 'test', transform=None)

        # 尝试不同的方式获取数据和标签
        if hasattr(cifar_dataobj, 'data') and hasattr(cifar_dataobj, 'targets'):
            data = np.array(cifar_dataobj.data)
            target = np.array(cifar_dataobj.targets)
        elif hasattr(cifar_dataobj, 'images') and hasattr(cifar_dataobj, 'labels'):
            data = np.array(cifar_dataobj.images)
            target = np.array(cifar_dataobj.labels)
        else:
            # 如果以上属性都不存在，需要通过迭代获取
            data = []
            target = []
            for i in range(len(cifar_dataobj)):
                img, label = cifar_dataobj[i]
                data.append(np.array(img))
                target.append(label)
            data = np.array(data)
            target = np.array(target)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # PaddlePaddle的FashionMNIST数据集
        fmnist_dataobj = FashionMNIST(self.root, mode='train' if self.train else 'test', transform=None)

        # 尝试不同的方式获取数据和标签
        if hasattr(fmnist_dataobj, 'data') and hasattr(fmnist_dataobj, 'targets'):
            data = np.array(fmnist_dataobj.data)
            target = np.array(fmnist_dataobj.targets)
        elif hasattr(fmnist_dataobj, 'images') and hasattr(fmnist_dataobj, 'labels'):
            data = np.array(fmnist_dataobj.images)
            target = np.array(fmnist_dataobj.labels)
        else:
            # 如果以上属性都不存在，需要通过迭代获取
            data = []
            target = []
            for i in range(len(fmnist_dataobj)):
                img, label = fmnist_dataobj[i]
                data.append(np.array(img))
                target.append(label)
            data = np.array(data)
            target = np.array(target)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolder_CINIC10(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


class ImageFolder_HAM10000(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

