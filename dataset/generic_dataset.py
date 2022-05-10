import os
import hydra
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from . import transforms as T
from torchvision.transforms import functional as F
import utils.io as io
import random


bbox_prewords = ['bound', 'locate', 'find']
bbox_postwords = ['with box', 'in rectangle']
mask_prewords = ['segment', 'separate', 'split']
mask_postwords = ['with mask', 'in mask']
depth_queries = ['estimate depth map of the image', 'generate the depth estimation']


def make_coco_transforms(image_set, high_resolution, cautious):
    if high_resolution:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        resize_before_crop = [400, 500, 600]
        crop_size = 384
    else:
        scales = [256]
        max_size = 384
        resize_before_crop = [256, 288, 320]
        crop_size = 224
    
    if image_set == "train":
        horizontal = [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(resize_before_crop),
                            T.RandomSizeCrop(crop_size, max_size, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                T.ToTensor(),
            ]
        )

    else:
        return T.Compose(
            [
                T.RandomResize([scales[-1]], max_size=max_size),
                T.ToTensor(),
            ]
        )


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
            return img, target.flatten() / ori_W
        else:
            # dense
            return img, F.resize(target, 256) 
    else:
        longer, shorter = max(H, W), min(H, W)
        if torch.numel(target) == 4:
            # bbox, determine which crop by distance along the longer side
            # input bbox labels are absolute values (int type)
            target = target.flatten() / ori_W * W

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
        
        if crop_flag == 0:
            return F.crop(img, 0, 0, shorter, shorter), target
        elif crop_flag == 1:
            return F.crop(img, (H-shorter)//2, (W-shorter)//2, shorter, shorter), target
        else:
            return F.crop(img, H-shorter, W-shorter, shorter, shorter), target


class GenericDataset(Dataset):
    def __init__(self, dataset_name, info, subset, task, online):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.subset = subset
        self.task = task
        self.samples = io.load_json_object(
            os.path.join(info['anno_dir'], f'{subset}.json')
        )
        self.online = online
        print(f'load {len(self.samples)} samples in {self.dataset_name}_{self.subset} in {"online" if online else "offline"} mode')
        if online:
            if task == 'depth':
                self.transform = make_coco_transforms(subset, high_resolution=False, cautious=False)
            else:
                self.transform = make_coco_transforms(subset, high_resolution=False, cautious=True)
    
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert("RGB")
        img = F.to_tensor(img)
        return img
    
    def read_image_pil(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert("RGB")
        return img

    def get_mask(self, segment_id):
        mask = Image.open(os.path.join(self.info.mask_dir, f'{segment_id}.png'))
        mask = F.to_tensor(mask)
        return mask

    def get_depth(self, depth_name):
        depth = Image.open(os.path.join(self.info.depth_dir, depth_name))
        depth = F.to_tensor(depth).float()
        return depth

    def depth_process(self, depth, mode):
        if 'taskonomy' in self.dataset_name:
            valid_mask = (depth < 40000).squeeze()
            if mode == 'log':
                # taskonomy, 2018
                depth = torch.log2(1.0+depth) / 16
            elif mode == 'linear':
                # X-TC, 2020
                depth = torch.clip(depth/8000, 0, 1)
            else:
                raise NotImplementedError
            return depth, valid_mask
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        sample = self.samples[i]

        if self.online:
            return self.get_online(sample, i)
        
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

        if self.task == 'depth':
            target = self.depth_process(target, mode='linear')

        targets = {
            'task': self.task,
            'target': target,
            'online': False,
            'buffer_name': self.get_encoding_fname(i)
        }
        return img, query, targets

    def get_online(self, sample, i):
        img = self.read_image_pil(sample['img_name'])

        if self.task == 'bbox':
            query = f'{random.choice(bbox_prewords)} "'+sample['sentences'][0]['sent']+f'" {random.choice(bbox_postwords)}'
            target = {
                'query': query,
                'boxes': torch.as_tensor(sample['bbox'], dtype=torch.float32).reshape(-1, 4)
            }
            img, target = self.transform(img, target)
            query = target['query']
            target = target['boxes']
        else:
            if self.task == 'mask':
                query = f'{random.choice(mask_prewords)} "'+sample['sentences'][0]['sent']+f'" {random.choice(mask_postwords)}'
                target = {
                    'query': query,
                    'boxes': torch.as_tensor(sample['bbox'], dtype=torch.float32).reshape(-1, 4),
                    'masks': self.get_mask(sample['segment_id'])
                }
                img, target = self.transform(img, target)
                query = target['query']
                target = target['masks']
            else:
                query = random.choice(depth_queries)
                target = {
                    'query': query,
                    'depth': self.get_depth(sample['depth_name'])
                }
                img, target = self.transform(img, target)
                query = target['query']
                target = target['depth'] 
            
        img, target = square_refine(img, target)

        if self.task == 'depth':
            target = self.depth_process(target, mode='linear')

        targets = {
            'task': self.task,
            'target': target,
            'online': True,
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
        depth = targets[0]['target'][0]
        print(depth.max(), depth.min())


if __name__=='__main__':
    main()
