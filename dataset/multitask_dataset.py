import hydra
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.misc import collate_fn as detr_collate_fn
from .generic_dataset import GenericDataset
from taming.vqgan import VQModel


class MultitaskDataset(Dataset):
    def __init__(self, datasets, subset, tasks=['bbox'], num_bins=200, vqgan=None):
        super().__init__()
        self.datasets = {}
        self.sample_l = []   # lower index
        self.sample_u = []   # upper index
        for dataset, info in datasets.items():
            for task in tasks:
                dataset_name = f'{dataset}_{task}'
                self.datasets[dataset_name] = GenericDataset(
                    dataset_name, info, subset, task, num_bins, vqgan
                )
                L = len(self.datasets[dataset_name])
                if len(self.sample_l) == 0:
                    self.sample_l.append(0)
                    self.sample_u.append(L)
                else:
                    self.sample_l.append(self.sample_u[-1])
                    self.sample_u.append(self.sample_u[-1]+L)

        self.sample_l = np.array(self.sample_l)
        self.sample_u = np.array(self.sample_u)
        # convert to numpy.array for advanced indexing
        self.dataset_names = list(self.datasets.keys())

    def __len__(self):
        N = 0
        for dataset_name, dataset in self.datasets.items():
            N += len(dataset)
        
        return N

    def __getitem__(self, i):
        cond = (i>=self.sample_l) * (i<self.sample_u)
        dataset_idx = cond.tolist().index(True)
        dataset_name = self.dataset_names[dataset_idx]
        return self.datasets[dataset_name][i-self.sample_l[dataset_idx]]

    def get_collate_fn(self):
        return detr_collate_fn
                
    def get_dataloader(self, **kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../config', config_name="vito.yaml")
def main(cfg):
    dataset = MultitaskDataset(cfg.learning.bbox.dataset, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        pass


if __name__=='__main__':
    main()
