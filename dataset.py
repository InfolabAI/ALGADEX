import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

dataset_stat_path = 'dataset_stat'

# transform class to conver 1 channel to 3 channel
class GrayToRGB:
    def __call__(self, image):
        return image.convert('RGB')

transform = transforms.Compose([GrayToRGB(), transforms.Resize(32), transforms.ToTensor()])
# decorator code to save mean and std of dataset for get_std() like the code in get_mean()
def save_stat(func):
    def wrapper(dataset, name):
        name_path = os.path.join(dataset_stat_path, name+".npz")

        if os.path.exists(name_path):
            # code to load npz
            ret = np.load(name_path)
            return [ret['arr_0'][0], ret['arr_0'][1], ret['arr_0'][2]]

        ret = func(dataset, name)
        # code to save npz
        np.savez(name_path, ret)

        return ret

    return wrapper


# 채널 별 mean 계산
@save_stat
def get_mean(dataset, name):
    print(f'Calculating mean of dataset {dataset}...')
    meanRGB = [np.mean(image.numpy(), axis=(1, 2)) for image, _ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]

# 채널 별 str 계산
@save_stat
def get_std(dataset, name):
    print(f'Calculating std of dataset {dataset}...')
    stdRGB = [np.std(image.numpy(), axis=(1, 2)) for image, _ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]

def get_MNIST():
    raw_tr = datasets.MNIST('/mnt/share_nfs/dataset', train=True, download=True, transform=transform)
    raw_te = datasets.MNIST('/mnt/share_nfs/dataset', train=False, download=False, transform=transform)
    raw_tr.transform = transforms.Compose(raw_tr.transform.transforms + [transforms.Normalize(get_mean(raw_tr, "mnist_tr_mean"),get_std(raw_tr, "mnist_tr_std"))])
    raw_te.transform = transforms.Compose(raw_te.transform.transforms + [transforms.Normalize(get_mean(raw_te, "mnist_te_mean"),get_std(raw_te, "mnist_te_std"))])
    return raw_tr, raw_te

def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('/mnt/share_nfs/dataset', train=True, download=True, transform=transform)
    raw_te = datasets.FashionMNIST('/mnt/share_nfs/dataset', train=False, download=True, transform=transform)
    raw_tr.transform = transforms.Compose(raw_tr.transform.transforms + [transforms.Normalize(get_mean(raw_tr, "fashionmnist_tr_mean"),get_std(raw_tr, "fashionmnist_tr_std"))])
    raw_te.transform = transforms.Compose(raw_te.transform.transforms + [transforms.Normalize(get_mean(raw_te, "fashionmnist_te_mean"),get_std(raw_te, "fashionmnist_te_std"))])
    return raw_tr, raw_te

def get_SVHN():
    raw_tr = datasets.SVHN('/mnt/share_nfs/dataset/svhn', split='train', download=True, transform=transform)
    raw_te = datasets.SVHN('/mnt/share_nfs/dataset/svhn', split='test', download=True, transform=transform)
    raw_tr.transform = transforms.Compose(raw_tr.transform.transforms + [transforms.Normalize(get_mean(raw_tr, "svhn_tr_mean"),get_std(raw_tr, "svhn_tr_std"))])
    raw_te.transform = transforms.Compose(raw_te.transform.transforms + [transforms.Normalize(get_mean(raw_te, "svhn_te_mean"),get_std(raw_te, "svhn_te_std"))])
    return raw_tr, raw_te

def get_CIFAR10():
    raw_tr = datasets.CIFAR10('/mnt/share_nfs/dataset/cifar10', train=True, download=True, transform=transform)
    raw_te = datasets.CIFAR10('/mnt/share_nfs/dataset/cifar10', train=False, download=True, transform=transform)
    raw_tr.transform = transforms.Compose(raw_tr.transform.transforms + [transforms.Normalize(get_mean(raw_tr, "cifar10_tr_mean"),get_std(raw_tr, "cifar10_tr_std"))])
    raw_te.transform = transforms.Compose(raw_te.transform.transforms + [transforms.Normalize(get_mean(raw_te, "cifar10_te_mean"),get_std(raw_te, "cifar10_te_std"))])
    return raw_tr, raw_te

def get_Imagenet():
    mean = [0.485, 0.456, 0.406]  # NOTE ImageNet mean, std
    std = [0.229, 0.224, 0.225]  # NOTE ImageNet mean, std
    raw_tr = datasets.ImageFolder('/mnt/share_nfs/imagenet/train_folder/', transform=transform)
    raw_te = datasets.ImageFolder('/mnt/share_nfs/imagenet/val_folder/', transform=transform)
    raw_tr.transform = transforms.Compose(raw_tr.transform.transforms + [transforms.Normalize(mean, std)])
    raw_te.transform = transforms.Compose(raw_te.transform.transforms + [transforms.Normalize(mean, std)])

    return raw_tr, raw_te

def get_dataset(name):
    os.makedirs(dataset_stat_path, exist_ok=True)

    if name == 'MNIST':
        return get_MNIST()
    elif name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()
    elif name == 'Imagenet':
        return get_Imagenet()

def get_stat(raw, name):
    mean = raw[0][0].mean().item()
    std = raw[0][0].std().item()
    max = raw[0][0].max().item()
    min = raw[0][0].min().item()
    print(f"{name} mean: {mean}, std: {std}, max: {max}, min: {min}")



if __name__ == '__main__':
    raw_tr, raw_te =    get_MNIST()
    get_stat(raw_te, 'MNIST')
    # MNIST mean: -0.13240988552570343, std: 0.852272093296051, max: 2.8567583560943604, min: -0.4363902807235718
    raw_tr, raw_te =    get_FashionMNIST()
    get_stat(raw_te, 'FashionMNIST')
    # FashionMNIST mean: -0.37360283732414246, std: 0.8424383997917175, max: 2.2295382022857666, min: -0.896783173084259
    raw_tr, raw_te =    get_SVHN()
    get_stat(raw_te, 'SVHN')
    # SVHN mean: -1.6991950273513794, std: 0.9394116997718811, max: 0.2940393388271332, min: -3.6624090671539307
    raw_tr, raw_te =    get_CIFAR10()
    get_stat(raw_te, 'CIFAR10')
    # CIFAR10 mean: -0.25679007172584534, std: 0.9096104502677917, max: 2.6753218173980713, min: -2.1953701972961426
    raw_tr, raw_te =    get_Imagenet()
    get_stat(raw_te, 'Imagenet')
    # Imagenet mean: 0.12454724311828613, std: 1.5195982456207275, max: 2.640000104904175, min: -2.1179039478302
