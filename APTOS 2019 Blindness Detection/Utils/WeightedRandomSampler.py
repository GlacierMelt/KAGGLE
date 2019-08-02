import torch
import torch.utils.data
import torchvision
from tqdm import tqdm_notebook as tqdm

class WeightedRSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, replacement=True):
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.replacement = replacement

        label_to_count = {}
        for idx in tqdm(range(self.num_samples)):
            label = self._get_label(self.dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[self._get_label(self.dataset, idx)]
                   for idx in range(self.num_samples)]
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.ImageFolder:
            return self.dataset.imgs[idx][1]
        else:
            return self.dataset[idx]['label']
