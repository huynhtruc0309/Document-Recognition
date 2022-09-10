import cv2
import torch
import random
import numpy as np

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset


class MatchingDataset(Dataset):
    def __init__(
        self,
        dirname,
        classes,
        image_size,
        image_patterns,
        device,
        return_triplets,
        num_channel,
        get_triplets_func,
        transforms=None,
    ):
        super(MatchingDataset, self).__init__()
        self.classes = classes
        self.idx_to_class = {idx_class: _class for _class, idx_class in classes.items()}
        self.image_size = image_size
        self.device = device
        self.return_triplets = return_triplets
        self.num_channel = num_channel
        self.transforms = transforms if transforms is not None else {}
        self.prev_transforms = self.transforms.get("prev_transforms", [])
        self.main_transforms = self.transforms.get("main_transforms", [])
        self.post_transforms = self.transforms.get("post_transforms", [])

        data_pairs = []
        for image_pattern in image_patterns:
            for image_path in Path(dirname).glob("**/{}".format(image_pattern)):
                target = image_path.parent.stem
                if target in self.classes:
                    data_pairs.append([image_path, self.classes[target]])

        self.data_pairs = natsorted(data_pairs, key=lambda x: str(x[0].stem))
        self.targets = list(np.asarray(self.data_pairs)[:, 1])
        self.triplets = [] if not self.return_triplets else get_triplets_func(self.targets)
        print(f"{Path(dirname).stem}: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs) if not self.return_triplets else len(self.triplets)

    def __getitem__(self, index):
        if not self.return_triplets:
            return self._get_pair(index=index)
        else:
            return self._get_triplet(index=index)

    def _get_pair(self, index):
        image_path, target = self.data_pairs[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # prev_transform
        for transform in self.prev_transforms:
            image = transform(image=image)

        # main transforms
        for transform in random.sample(
            self.main_transforms, k=random.randint(0, len(self.main_transforms))
        ):
            image = transform(image=image)

        # post_transform
        for transform in self.post_transforms:
            image = transform(image=image)

        # gray scale image
        if self.num_channel == 1:
            image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)

        # image
        sample = torch.from_numpy(image)
        sample = torch.div(sample, 255.0)
        sample = (
            sample.permute(2, 0, 1)
            .contiguous()
            .to(device=self.device, dtype=torch.float32)
        )

        # convert to Torch Tensor
        target = torch.tensor(target).to(device=self.device, dtype=torch.float32)

        return sample, target, str(image_path)

    def _get_triplet(self, index):
        idx_anchor, idx_positive, idx_negative = self.triplets[index]

        # load images
        anchor_path, _ = self.data_pairs[idx_anchor]
        anchor = cv2.imread(str(anchor_path))
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)

        positive_path, _ = self.data_pairs[idx_positive]
        positive = cv2.imread(str(positive_path))
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)

        negative_path, _ = self.data_pairs[idx_negative]
        negative = cv2.imread(str(negative_path))
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        # prev_transform
        for transform in self.prev_transforms:
            anchor = transform(image=anchor)
            positive = transform(image=positive)
            negative = transform(image=negative)

        # main transform
        for transform in random.sample(
            self.main_transforms, k=random.randint(0, len(self.main_transforms))
        ):
            anchor = transform(image=anchor)
            positive = transform(image=positive)
            negative = transform(image=negative)

        # post_transform
        for transform in self.post_transforms:
            anchor = transform(image=anchor)
            positive = transform(image=positive)
            negative = transform(image=negative)

        # gray scale image
        if self.num_channel == 1:
            anchor = np.expand_dims(cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY), axis=-1)
            positive = np.expand_dims(cv2.cvtColor(positive, cv2.COLOR_RGB2GRAY), axis=-1)
            negative = np.expand_dims(cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY), axis=-1)

        # convert to tensor
        anchor = torch.from_numpy(anchor)
        anchor = torch.div(anchor, 255.0)
        anchor = (
            anchor.permute(2, 0, 1)
            .contiguous()
            .to(device=self.device, dtype=torch.float32)
        )

        positive = torch.from_numpy(positive)
        positive = torch.div(positive, 255.0)
        positive = (
            positive.permute(2, 0, 1)
            .contiguous()
            .to(device=self.device, dtype=torch.float32)
        )

        negative = torch.from_numpy(negative)
        negative = torch.div(negative, 255.0)
        negative = (
            negative.permute(2, 0, 1)
            .contiguous()
            .to(device=self.device, dtype=torch.float32)
        )

        return anchor, positive, negative
