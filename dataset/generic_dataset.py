import os
import hydra
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import utils.io as io
import random


bbox_prewords = ['bound', 'locate', 'find']
bbox_postwords = ['with box', 'in rectangle']
mask_prewords = ['segment', 'separate', 'split']
mask_postwords = ['with mask', 'in mask']
depth_queries = ['estimate depth map of the image', 'generate the depth estimation']


def square_refine(img, target):
    """
    make corresponding refinement on img to match square cropping
    img: [3, H, W]
    crop_flag: 0 for left(up), 1 for center, 2 for right(down)
    return img in square shape and corresponding target
    """
    ori_W = img.shape[-1]
    img = F.resize(img, 256)
    H, W = img.shape[-2:]

    if H == W:
        if torch.numel(target) == 4:
            # bbox
            return img, target / ori_W
        else:
            # dense
            return img, F.resize(target, 256) 
    else:
        longer, shorter = max(H, W), min(H, W)
        if torch.numel(target) == 4:
            # bbox, determine which crop by distance along the longer side
            # input bbox labels are absolute values (int type)
            target = target / ori_W * W

            bbox_center = (target[0]+target[2])/2 if W > H else (target[1]+target[3])/2
            crop_centers = torch.tensor([shorter/2, longer/2, longer-shorter/2])
            crop_flag = torch.argmin(torch.abs(bbox_center-crop_centers))
            # update bbox
            modify_axis = 1 if H > W else 0
            if crop_flag == 0:
                target[modify_axis::2] = target[modify_axis::2]
            elif crop_flag == 1:
                target[modify_axis::2] = (target[modify_axis::2]-(longer-shorter)/2)
            else:
                target[modify_axis::2] = (target[modify_axis::2]-(longer-shorter))
            
            target = torch.clip(target/shorter, 0, 1)
        else:
            # dense, determine which crop by summation of values on cropped region
            target = F.resize(target, 256)

            candidates = torch.stack([
                target[..., :shorter, :shorter],
                target[..., (H-shorter)//2:(H+shorter)//2, (W-shorter)//2:(W+shorter)//2],
                target[..., -shorter:, -shorter:]
            ], dim=0)
            crop_flag = torch.argmax(torch.sum(candidates, dim=(1,2,3)))
            target = candidates[crop_flag]
            
            target = target / torch.max(target)
        
        if crop_flag == 0:
            return F.crop(img, 0, 0, shorter, shorter), target
        elif crop_flag == 1:
            return F.crop(img, (H-shorter)//2, (W-shorter)//2, shorter, shorter), target
        else:
            return F.crop(img, H-shorter, W-shorter, shorter, shorter), target


class GenericDataset(Dataset):
    def __init__(self, dataset_name, info, subset, task):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.subset = subset
        self.task = task
        self.samples = io.load_json_object(
            os.path.join(info['anno_dir'], f'{subset}.json')
        )
        print(f'load {len(self.samples)} samples in {self.dataset_name}_{self.subset}')
    
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert("RGB")
        img = F.to_tensor(img)
        return img

    def get_mask(self, segment_id):
        mask = Image.open(os.path.join(self.info.mask_dir, f'{segment_id}.png'))
        mask = F.to_tensor(mask)
        return mask

    def get_depth(self, depth_name):
        depth = Image.open(os.path.join(self.info.depth_dir, depth_name))
        depth = F.to_tensor(depth)
        return depth / torch.max(depth)

    def __getitem__(self, i):
        sample = self.samples[i]
        img = self.read_image(sample['img_name'])

        if self.task == 'bbox':
            query = f'{random.choice(bbox_prewords)} "'+sample['sentences'][0]['sent']+f'" {random.choice(bbox_postwords)}'
            target = torch.as_tensor(sample['bbox'], dtype=torch.float32)
        elif self.task == 'mask':
            query = f'{random.choice(mask_prewords)} "'+sample['sentences'][0]['sent']+f'" {random.choice(mask_postwords)}'
            target = self.get_mask(sample['segment_id'])
        elif self.task == 'depth':
            query = random.choice(depth_queries)
            target = self.get_depth(sample['depth_name'])
        else:
            raise NotImplementedError

        img, target = square_refine(img, target)

        targets = {
            'task': self.task,
            'target': target,
            'buffer_name': self.get_encoding_fname(i)
        }
        return img, query, targets
                
    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

    def get_encoding_fname(self, i):
        return f'{self.dataset_name}_{self.subset}_{i}.pt'


def collate_fn(batch):
        batch = list(zip(*batch))
        return tuple(batch)


@hydra.main(config_path="../config", config_name="vito.yaml")
def main(cfg):
    dataset = GenericDataset('taskonomy_depth', cfg.dataset.depth.taskonomy, 'val', 'depth')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, queries, targets = data
        depth = targets[0]['target']
        print(depth.max(), depth.min())


if __name__=='__main__':
    main()
