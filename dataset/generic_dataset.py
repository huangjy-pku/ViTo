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
from taming.vqgan import VQModel


def bbox_process(bbox, num_bins):
    """
    from absolute coordinates to position bins token
    input bbox: [x_min, y_min, x_max, y_max]
    return as string splitting tokens
    """
    x1, y1, x2, y2 = bbox
    token_x1 = np.clip(round(float(x1)*num_bins-0.5), 0, num_bins-1)
    token_y1 = np.clip(round(float(y1)*num_bins-0.5), 0, num_bins-1)
    token_x2 = np.clip(round(float(x2)*num_bins-0.5), 0, num_bins-1)
    token_y2 = np.clip(round(float(y2)*num_bins-0.5), 0, num_bins-1)
    return f'__bbox_begin__ pos_{str(token_x1)} pos_{str(token_y1)} pos_{str(token_x2)} pos_{str(token_y2)} __bbox_end__'


def dense_process(mask_ori, vqgan):
    """
    from dense binary mask to token sequence
    original mask: tensor at shape [1, H, W], value in {0.0, 1.0}
    vqgan requires input at shape [3, 256, 256], value in {-1, 1}
    vqgan encoder outputs token sequence [8x8] represented in indices, value in [0, 1023]
    finally convert into string
    return mask at shape [1, 256, 256], value in {0.0, 1.0}
    crop_flag: 0 for left(up), 1 for center, 2 for right(down)
    """
    # resize to 256x256, scale first then crop center
    mask_ori = T.F.resize(mask_ori, 256)
    H, W = mask_ori.shape[-2:]
    mask = T.F.crop(mask_ori, (H-256)//2, (W-256)//2, 256, 256)
    crop_flag = 1
    # try center crop, and make sure target present
    if torch.all(mask==0):
        mask = T.F.crop(mask_ori, 0, 0, 256, 256)
        crop_flag = 0
    if torch.all(mask==0):
        mask = T.F.crop(mask_ori, H-256, W-256, 256, 256)
        crop_flag = 2
    if torch.all(mask=0):
        raise Exception("empty mask encountered")

    # value rescale to [-1, 1]
    mask_vqgan = 2*mask - 1

    # replicate three channels
    mask_vqgan = mask_vqgan.repeat(3, 1, 1)

    # encode mask to sequence
    with torch.no_grad():
        encoding_indices = vqgan.encode(mask_vqgan)[-1][-1]
    # length 8x8, value in [0, 1023]
    
    target_seq = '__dense_begin__'
    for idx in encoding_indices:
        target_seq.append(f' code_{str(idx)}')
    target_seq.append(' __dense_end__')

    return target_seq, mask, crop_flag


def img_refine_square(img, crop_flag):
    """
    make corresponding refinement on img to match mask cropping
    img: [3, H, W]
    crop_flag: 0 for left(up), 1 for center, 2 for right(down)
    return img in square shape
    """
    H, W = img.shape[-2:]
    shorter = min(H, W)
    if crop_flag == 1:
        return T.F.crop(img, (H-shorter)//2, (W-shorter)//2, shorter, shorter)
    elif crop_flag == 0:
        return T.F.crop(img, 0, 0, shorter, shorter)
    elif crop_flag == 2:
        return T.F.crop(img, H-shorter, W-shorter, shorter, shorter)


def make_coco_transforms(image_set, high_resolution, cautious):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if high_resolution:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        resize_before_crop = [400, 500, 600]
        crop_size = 384
    else:
        scales = [224, 256]
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
                normalize,
            ]
        )

    elif image_set == "val" or 'test' in image_set:
        return T.Compose(
            [
                T.RandomResize([scales[-1]], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


class GenericDataset(Dataset):
    def __init__(self, dataset_name, info, subset, task, num_bins=200, vqgan_cfg=None):
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
        self.transforms = make_coco_transforms(subset, high_resolution=True, cautious=True)
        if task == 'dense':
            self.vqgan = VQModel(ddconfig=vqgan_cfg.ddconfig, n_embed=vqgan_cfg.n_embed,
                        embed_dim=vqgan_cfg.embed_dim, ckpt_path=vqgan_cfg.ckpt)
            self.vqgan.to(vqgan_cfg.device)
            self.vqgan.eval()
    
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
            query = 'bound '+sample['sentences'][0]['sent']+' with box.'
            target = {
                'query': query,
                'boxes': bbox
            }
        elif self.task == 'dense':
            query = 'segment '+sample['sentences'][0]['sent']+' with mask.'
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
            mask = target['mask']
            target_seq, mask, crop_flag = dense_process(mask, self.vqgan)
            targets.update({
                'mask': mask,
                'answer': target_seq
            })
            img = img_refine_square(img, crop_flag)
        else:
            raise NotImplementedError

        return img, query, targets, \
            self.get_encoding_fname(i)
    
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

    def get_encoding_fname(self, i):
        return f'{self.dataset_name}_{self.subset}_{i}.pt'


@hydra.main(config_path="../config", config_name="vito.yaml")
def main(cfg):
    dataset = GenericDataset(cfg.dataset.refcoco, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        pass

if __name__=='__main__':
    main()
