import os
import os.path
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


class CIFAR10_truncated:
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        if self.train:
            data = x_train
            target = y_train.flatten()
        else:
            data = x_test
            target = y_test.flatten()
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


class CIFAR100_truncated:
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        if self.train:
            data = x_train
            target = y_train.flatten()
        else:
            data = x_test
            target = y_test.flatten()
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


class FashionMNIST_truncated:
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        if self.train:
            data = x_train
            target = y_train
        else:
            data = x_test
            target = y_test
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            if len(self.data.shape) == 3:
                self.data[gs_index, :, :] = 0.0
            else:
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


class ImageFolder_CINIC10:
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Use tf.keras.utils.image_dataset_from_directory or manual loading
        import glob
        import os
        self.samples = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                for img_file in glob.glob(os.path.join(class_path, '*')):
                    class_idx = int(class_dir) if class_dir.isdigit() else 0
                    self.samples.append((img_file, class_idx))
        self.samples = np.array(self.samples)
        if self.dataidxs is not None:
            self.samples = self.samples[self.dataidxs]
            
    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageFolder_HAM10000:
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Use manual loading similar to ImageFolder_CINIC10
        import glob
        import os
        self.samples = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                for img_file in glob.glob(os.path.join(class_path, '*')):
                    class_idx = int(class_dir) if class_dir.isdigit() else 0
                    self.samples.append((img_file, class_idx))
        self.samples = np.array(self.samples)
        if self.dataidxs is not None:
            self.samples = self.samples[self.dataidxs]

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)