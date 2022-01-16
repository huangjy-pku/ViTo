import os
import hydra
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from . import transforms as T

import utils.io as io
from utils.misc import collate_fn as detr_collate_fn


def bbox_process(bbox, num_bins):
    """
    from absolute coordinates to position bins token
    input bbox: [x_min, y_min, x_max, y_max]
    return as tokens
    """
    x1, y1, x2, y2 = bbox
    token_x1 = np.clip(round(float(x1)*num_bins-0.5), 0, num_bins-1)
    token_y1 = np.clip(round(float(y1)*num_bins-0.5), 0, num_bins-1)
    token_x2 = np.clip(round(float(x2)*num_bins-0.5), 0, num_bins-1)
    token_y2 = np.clip(round(float(y2)*num_bins-0.5), 0, num_bins-1)
    return f'__bbox_begin__ pos_{str(token_x1)} pos_{str(token_y1)} pos_{str(token_x2)} pos_{str(token_y2)} __bbox_end__'


def dense_process(mask):
    """
    from dense binary mask to token sequence
    """
    return mask


def make_coco_transforms(image_set, cautious):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    max_size = 1333
    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, max_size, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


class GenericDataset(Dataset):
    def __init__(self, dataset_name, info, subset, task, num_bins=200):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.subset = subset
        self.task = task
        self.num_bins = num_bins
        self.samples = io.load_json_object(
            os.path.join(info.anno_dir, f'{subset}.json')
        )
        print(f'load {len(self.samples)} samples in {self.dataset_name}_{self.subset}')
        self.transforms = make_coco_transforms(subset, cautious=False)
    
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):    
        return Image.open(os.path.join(self.info.img_dir, img_name)).convert("RGB")

    def get_mask(self, segment_id):
        mask = Image.open(os.path.join(self.info.mask_dir, f'{segment_id}.png'))
        mask = F.to_tensor(mask)
        return mask

    def __getitem__(self, i):
        sample = self.samples[i]

        img = self.read_image(sample['img_name'])
        bbox = torch.as_tensor(sample['bbox'], dtype=torch.float32).reshape(-1, 4)
        mask = self.get_mask(sample['segment_id'])

        if self.task == 'bbox':
            query = 'bound '+sample['sentences'][0]['sent']+' with box'
            target = {
                'query': query,
                'boxes': bbox
            }
        elif self.task == 'dense':
            query = 'segment '+sample['sentences'][0]['sent']+' with mask'
            target = {
                'query': query,
                'masks': mask
            }
        else:
            raise NotImplementedError

        img, target = self.transforms(img, target)

        targets = {'task': self.task}

        query = target['query']

        if 'boxes' in target:
            bbox = target['boxes'][0]
            target_seq = bbox_process(bbox, self.num_bins)
            targets.update({
                'bbox': bbox,
                'answer': target_seq
            })
        elif 'masks' in target:
            mask = target['mask'][0]
            target_seq = dense_process(mask)
            targets.update({
                'mask': mask,
                'answer': target_seq
            })
        else:
            raise NotImplementedError

        return img, query, targets, \
            self.get_encoding_fname(self.dataset_name, self.subset, i)
    
    def get_images_from_tensor(self, imgs):
        """
        imgs: nested tensor: [images, masks]
        """
        device = imgs.tensors.device
        masks = imgs.mask
        imgs = 255 * (
            torch.tensor([0.485, 0.456, 0.406]).cuda(device) + \
                torch.tensor([0.229, 0.224, 0.225]).cuda(device)*imgs.tensors.permute(0, 2, 3, 1)
        )
        # imgs: [batch_size, H, W, C]
        # mask: [batch_size, H, W], value is True if padded
        return imgs, masks

    def get_collate_fn(self):
        return detr_collate_fn
                
    def get_dataloader(self, **kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

    def get_encoding_fname(self, dataset_name, subset, i):
        return f'{dataset_name}_{subset}_{i}.pt'


@hydra.main(config_path="../config", config_name="vito.yaml")
def main(cfg):
    dataset = GenericDataset(cfg.dataset.refcoco, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        pass

if __name__=='__main__':
    main()
