import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_MNIST, get_CIFAR10, get_Imagenet
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
# get balanced sampler


class Subset(Dataset):
    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices
        self.targets = np.array(dataset.targets)[indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# Step 1: Define the class to combine two datasets
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.label_adjust = dataset1.targets.unique().shape[0]
        self.dataset2 = dataset2
        self.targets = np.concatenate([np.array(dataset1.targets), np.array(dataset2.targets) + self.label_adjust])

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx][0], self.targets[idx]
        else:
            img, _ = self.dataset2[idx - len(self.dataset1)]
            return img, self.targets[idx]

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

# Step 2: Define the class for active learning with labeled indices and distractor indices (normal mode)
class ActiveLearningNormalDataset(Dataset):
    def __init__(self, combined_dataset, labeled_indices, distractor_class_idx):
        self.combined_dataset = combined_dataset
        self.labeled_indices = labeled_indices
        self.set_distractor_indices_per_class(distractor_class_idx)
    
    def set_distractor_indices_per_class(self, class_idx):
        # class idx 보다 크거나 같은 모든 class 에 대해 distractor indices 로 설정
        self.distractor_indices = sorted(list(set([i for i in range(len(self.combined_dataset)) if self.combined_dataset.targets[i] >= class_idx])))
        self.stage_indices = sorted(list(set(self.labeled_indices) - set(self.distractor_indices)))

    def update_labeled_indices(self, new_indices):
        self.labeled_indices.extend(new_indices)
        self.stage_indices = sorted(list(set(self.labeled_indices) - set(self.distractor_indices)))

    def __getitem__(self, idx):
        actual_idx = list(self.stage_indices)[idx]
        return self.combined_dataset[actual_idx]

    def __len__(self):
        return len(self.stage_indices)
    
    def get_labels(self):
        return self.combined_dataset.targets[self.stage_indices]

# Step 3: Define the class for binary classification mode
class ActiveLearningBinaryDataset(Dataset):
    def __init__(self, combined_dataset, labeled_indices, distractor_class_idx):
        self.combined_dataset = combined_dataset
        self.labeled_indices = labeled_indices
        self.set_distractor_indices_per_class(distractor_class_idx)
    
    def set_distractor_indices_per_class(self, class_idx):
        # class idx 보다 크거나 같은 모든 class 에 대해 distractor indices 로 설정
        self.distractor_indices = sorted(list(set([i for i in range(len(self.combined_dataset)) if self.combined_dataset.targets[i] >= class_idx])))

    def update_labeled_indices(self, new_indices):
        self.labeled_indices.extend(new_indices)

    def __getitem__(self, idx):
        actual_idx = self.labeled_indices[idx]
        if actual_idx in self.distractor_indices:
            return self.combined_dataset[actual_idx][0], 1 # combined_dataset 의 label 을 무시하고, lavel 을 0, 1 로 변환
        else:
            return self.combined_dataset[actual_idx][0], 0

    def __len__(self):
        return len(self.labeled_indices)
    
    def get_labels(self):
        return np.array([1 if i in self.distractor_indices else 0 for i in self.labeled_indices])


if __name__ == '__main__':
    # NOTE TEST 코드

    raw_mn_tr, raw_mn_te = get_MNIST()
    raw_ig_tr, raw_ig_te = get_Imagenet()

    # Initialize the UnifiedDataset
    combined_dataset = CombinedDataset(raw_mn_te, raw_ig_te)

    # Set subset
    sub_combined_dataset = Subset(combined_dataset, list(range(len(combined_dataset)//2)))

    # Define initial labeled indices and distractor indices
    labeled_indices = [0, 1, 2, 3, 4, 5] + list(range(10,50))

    al_dataset = ActiveLearningNormalDataset(sub_combined_dataset, labeled_indices, 6)
    bcl_dataset = ActiveLearningBinaryDataset(sub_combined_dataset, labeled_indices, 6)

    # Test: Length in normal mode
    print("Length in normal mode:", len(al_dataset))  # Expected: 7 (labeled - distractors)

    # Test: Update labeled indices and check length
    new_indices = list(range(60, 90))
    al_dataset.update_labeled_indices(new_indices)
    print("Length after updating labeled indices in normal mode:", len(al_dataset))  # Expected: 11 (new labeled indices - distractors)

    # Test: Length and indexing in binary classification mode
    print("Length in binary classification mode:", len(bcl_dataset))  # Expected: 15 (labeled indices)

    # Test: Switching back to normal mode
    dataloader = DataLoader(al_dataset, batch_size=16, sampler=ImbalancedDatasetSampler(al_dataset), num_workers=2)
    for batch in dataloader:
        print("Batch in normal mode:", batch[0].shape, batch[1].unique())

    # Test: Iterating over DataLoader
    dataloader = DataLoader(bcl_dataset, batch_size=16, sampler=ImbalancedDatasetSampler(bcl_dataset), num_workers=2)
    for batch in dataloader:
        print("Batch in binary classification mode:", batch[0].shape, batch[1].unique())

    """
    이 테스트 코드는 다음을 검증합니다:
    1. 정상 모드에서 데이터셋의 길이와 인덱싱 기능
    2. 새로운 레이블링된 인덱스를 추가한 후 데이터셋의 길이
    3. 이진 분류 모드에서 데이터셋의 길이와 인덱싱 기능
    4. `DataLoader`를 사용하여 배치 단위로 데이터를 로드하는 기능
    """