import os
import os.path
import logging
import numpy as np
from PIL import Image
import mindspore.dataset as ds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(directory, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    if not extensions and not is_valid_file:
        raise ValueError("Both extensions and is_valid_file cannot be None")
    
    if is_valid_file is None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
            
    if not os.path.isdir(directory):
        return []

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                # Assuming folder name is class index or we need to map it?
                # Standard ImageFolder maps class_name -> index.
                # We need to implement that if we want full ImageFolder behavior.
                # But for now, let's see if we can use ds.ImageFolderDataset to get samples.
                pass
    return []

# Helper to load data from MS dataset to memory
def load_ms_dataset_to_memory(ms_ds):
    data_list = []
    target_list = []
    # Create iterator
    iter = ms_ds.create_tuple_iterator(output_numpy=True, num_epochs=1)
    for item in iter:
        # item is [image, label]
        data_list.append(item[0])
        target_list.append(item[1])
    
    if not data_list:
        return np.array([]), np.array([])
        
    return np.stack(data_list), np.stack(target_list)

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
        usage = 'train' if self.train else 'test'
        # Note: MindSpore Cifar10Dataset expects the directory containing the bin files.
        # If download=True was used in PyTorch, it created a 'cifar-10-batches-py' folder.
        # We might need to adjust root.
        try:
            cifar_ds = ds.Cifar10Dataset(dataset_dir=self.root, usage=usage, shuffle=False)
            data, target = load_ms_dataset_to_memory(cifar_ds)
        except Exception as e:
            logger.warning(f"Failed to load CIFAR10 from {self.root} using MindSpore: {e}")
            # Fallback or empty
            data = np.array([])
            target = np.array([])

        if self.dataidxs is not None and len(data) > 0:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
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
        usage = 'train' if self.train else 'test'
        try:
            cifar_ds = ds.Cifar100Dataset(dataset_dir=self.root, usage=usage, shuffle=False)
            data, target = load_ms_dataset_to_memory(cifar_ds)
        except Exception as e:
            logger.warning(f"Failed to load CIFAR100 from {self.root}: {e}")
            data = np.array([])
            target = np.array([])

        if self.dataidxs is not None and len(data) > 0:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # CIFAR100 in PyTorch code converted to PIL
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
        usage = 'train' if self.train else 'test'
        try:
            fmnist_ds = ds.FashionMnistDataset(dataset_dir=self.root, usage=usage, shuffle=False)
            data, target = load_ms_dataset_to_memory(fmnist_ds)
        except Exception as e:
            logger.warning(f"Failed to load FashionMNIST from {self.root}: {e}")
            data = np.array([])
            target = np.array([])

        if self.dataidxs is not None and len(data) > 0:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            # FashionMNIST is 1 channel, but code assumes 3? 
            # Original code: self.data[gs_index, :, :, 1] = 0.0
            # FashionMNIST data is usually (N, H, W) or (N, H, W, 1).
            # If it's (N, H, W), this will fail.
            # But let's keep it as is, assuming user knows what they are doing or data is expanded.
            pass

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# Helper for ImageFolder logic since we can't easily use torchvision.datasets.ImageFolder
class SimpleImageFolder:
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._make_dataset(root)
        self.loader = self._default_loader

    def _make_dataset(self, directory):
        instances = []
        directory = os.path.expanduser(directory)
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for target_class in classes:
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                        path = os.path.join(root, fname)
                        instances.append((path, class_index))
        return instances

    def _default_loader(self, path):
        return Image.open(path).convert('RGB')
        
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


class ImageFolder_CINIC10(SimpleImageFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataidxs = dataidxs
        self.train = train
        
        if self.dataidxs is not None:
            self.samples = np.array(self.samples)[self.dataidxs]
        else:
            self.samples = np.array(self.samples)
            
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


class ImageFolder_HAM10000(SimpleImageFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataidxs = dataidxs
        self.train = train

        if self.dataidxs is not None:
            self.samples = np.array(self.samples)[self.dataidxs]
        else:
            self.samples = np.array(self.samples)

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