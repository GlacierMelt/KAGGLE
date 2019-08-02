import math
import torch
import torch.utils.data
import torchvision
import torch.distributed as dist

class W_DistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset,  num_replicas=None, rank=None, replacement=True):
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement

        # WeightRandom
        label_to_count = {}
        for idx in range(len(self.dataset)):
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in range(len(self.dataset))]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        indices = torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.ImageFolder:
            return self.dataset.imgs[idx][1]
        else:
            return self.dataset[idx]['label']
