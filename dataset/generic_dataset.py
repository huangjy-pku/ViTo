import os
import hydra
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from . import transforms as T
from random import randint

import utils.io as io
from utils.misc import collate_fn as detr_collate_fn

clef_category = ['', '', 'aerostatic balloon', 'air vehicles', 'airplane', 'ancent building', 'animal', 'ant', 'antelope',
                 'ape', 'apple', 'arctic', 'astronaut', 'baby', 'ball', 'balloon', 'beach', 'bear', 'beaver', 'bed', 'beetle',
                 'bench', 'bicycle', 'bird', 'boat', 'boat rafting', 'bobcat wildcat', 'book', 'bottle', 'branch', 'bridge',
                 'building', 'bull', 'bus', 'bush', 'butterfly', 'cabin', 'cactus', 'camel', 'camera', 'can', 'canine',
                 'cannon', 'car', 'caribou', 'castle', 'cat', 'caterpillar', 'cello', 'chair', 'cheetah', 'child', 'child boy',
                 'child girl', 'chimney', 'church', 'church interior', 'city', 'clock', 'cloth', 'cloud', 'column', 'construction',
                 'construction other', 'coral', 'cougar puma', 'couple of persons', 'cow', 'coyote', 'crab', 'crocodile', 'cup',
                 'curtain', 'deer', 'desk', 'dessert', 'dish', 'diver', 'dog', 'dolphin', 'door', 'dragonfly', 'eagle', 'edifice',
                 'elephant', 'elk', 'entity', 'fabric', 'face of person', 'feline', 'fence', 'fire', 'firework', 'fish', 'flag',
                 'flamingo', 'flock of birds', 'floor', 'floor carpet', 'floor other', 'floor tennis court', 'floor wood', 'flower',
                 'flowerbed', 'food', 'fountain', 'fowl hen', 'fox', 'fruit', 'furniture', 'furniture other', 'generic objects',
                 'giraffe', 'glacier', 'glass', 'goat', 'grapes', 'grass', 'ground', 'ground vehicles', 'group of persons', 'guitar',
                 'hand of person', 'handcraft', 'hat', 'hawk', 'head of person', 'hedgehog porcupine', 'helicopter', 'herd of mammals',
                 'highway', 'hill', 'horn', 'horse', 'house', 'humans', 'hut', 'ice', 'iguana', 'insect', 'island', 'jaguar', 'jewelry',
                 'kangaroo', 'kitchen pot', 'koala', 'lake', 'lamp', 'landscape nature', 'leaf', 'leopard', 'lighthouse', 'lion',
                 'lizard', 'llama', 'lobster', 'log', 'lynx', 'mammal', 'mammal other', 'man', 'man made', 'man made other', 'mandril',
                 'marsupial', 'monkey', 'monument', 'motorcycle', 'mountain', 'mural carving', 'mushroom', 'musical instrument',
                 'nest', 'non wooden furniture', 'ocean', 'ocean animal', 'octopus', 'orange', 'other entity', 'owl', 'pagoda',
                 'painting', 'palm', 'panda', 'paper', 'parrot', 'penguin', 'person', 'person related objects', 'piano', 'pigeon',
                 'plant', 'plant pot', 'polar bear', 'primate', 'public sign', 'pyramid', 'rabbit', 'rafter', 'railroad', 'reptile',
                 'rhinoceros', 'river', 'road', 'rock', 'rodent', 'roof', 'rooster', 'ruin archeological', 'sand beach', 'sand dessert',
                 'saxophone', 'school of fishes', 'scorpion', 'screen', 'seahorse', 'seal', 'semaphore', 'sheep', 'shell', 'ship',
                 'shore', 'sidewalk', 'sky', 'sky blue', 'sky light', 'sky night', 'sky red sunset dusk', 'smoke', 'snake', 'snow',
                 'space shuttle', 'squirrel', 'stairs', 'starfish', 'statue', 'steam', 'strawberry', 'street', 'sun', 'surfboard',
                 'swimming pool', 'table', 'telephone', 'tiger', 'tire', 'tower', 'toy', 'train', 'trash', 'tree', 'trees', 'trombone',
                 'trumpet', 'trunk', 'turtle', 'umbrella', 'vegetable', 'vegetation', 'vehicle', 'vehicles with tires', 'viola',
                 'violin', 'volcano', 'wall', 'water', 'water reflection', 'water vehicles', 'waterfall', 'waves', 'whale', 'window',
                 'wolf', 'woman', 'wood', 'wooden furniture', 'zebra']

coco_category = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']


def bbox_process(bbox, num_bins, cat, cat_type):
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
    if cat_type == 'refclef':
        target_seq = f'{clef_category[cat]} '
    elif cat_type == 'refcoco':
        target_seq = f'{coco_category[cat]} '
    return target_seq + f'__bbox_begin__ pos_{str(token_x1)} pos_{str(token_y1)} pos_{str(token_x2)} pos_{str(token_y2)} __bbox_end__'

def dense_process(mask_ori, vqgan, cat, cat_type, noseq=False):
    """
    from dense binary mask to token sequence
    original mask: tensor at shape [1, H, W], int value in {0.0, 1.0}
    vqgan requires input at shape [1, 3, 256, 256], float value in [-1, 1]
    vqgan encoder outputs token sequence [8x8] represented in indices, value in [0, 1023]
    finally convert into string
    return mask at shape [1, 256, 256], value in {0.0, 1.0}
    crop_flag: 0 for left(up), 1 for center, 2 for right(down)
    """
    # resize to 256x256, scale first then crop center
    mask_ori = T.F.resize(mask_ori, 256)
    H, W = mask_ori.shape[-2:]
    mask_0 = T.F.crop(mask_ori, 0, 0, 256, 256)
    mask_1 = T.F.crop(mask_ori, (H-256)//2, (W-256)//2, 256, 256)
    mask_2 = T.F.crop(mask_ori, H-256, W-256, 256, 256)
    crop_flag = np.argmax([torch.sum(mask_0), torch.sum(mask_1), torch.sum(mask_2)])
    # select the most occupation
    mask = [mask_0, mask_1, mask_2][crop_flag]
    if noseq:
        return None, mask, crop_flag

    # value rescale to [-1, 1]
    mask_vqgan = 2*mask - 1

    # replicate three channels
    mask_vqgan = mask_vqgan.repeat(3, 1, 1).unsqueeze(0).to(torch.float).to(vqgan.device)

    # encode mask to sequence
    with torch.no_grad():
        encoding_indices = vqgan.encode(mask_vqgan)[-1][-1]
    # length 8x8, value in [0, 1023]

    if cat_type == 'refclef':
        target_seq = f'{clef_category[cat]} '
    elif cat_type == 'refcoco':
        target_seq = f'{coco_category[cat]} '
    
    target_seq += '__dense_begin__'
    for idx in encoding_indices:
        target_seq += f' code_{str(idx.item())}'
    target_seq += ' __dense_end__'

    return target_seq, mask, crop_flag


def img_refine_square(img, crop_flag):
    """
    make corresponding refinement on img to match mask cropping
    img: [3, H, W] or [H, W]
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
    def __init__(self, dataset_name, info, subset, task, num_bins, vqgan, aug):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.subset = subset
        self.task = task
        self.samples = io.load_json_object(
            os.path.join(info.anno_dir, f'{subset}.json')
        )
        print(f'load {len(self.samples)} samples in {self.dataset_name}_{self.subset}')
        self.transforms = make_coco_transforms(subset, high_resolution=True, cautious=True)
        if task == 'bbox':
            self.num_bins = num_bins
        # elif task == 'dense':
        #     self.vqgan = vqgan
        self.augmentation = aug
        if aug == 'offline':
            self.offline_root = info.offline_root
    
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
        if self.augmentation == 'offline' and self.subset == 'train' and self.task == 'dense':
            return self.get_offline(sample, i)
        else:
            return *self.get_online(sample), self.get_encoding_fname(i)
    
    def get_online(self, sample):
        img = self.read_image(sample['img_name'])
        query = sample['sentences'][0]['sent']

        bbox = torch.as_tensor(sample['bbox'], dtype=torch.float32).reshape(-1, 4)
        mask = self.get_mask(sample['segment_id'])
        target = {
                'query': query,
                'boxes': bbox,
                'masks': mask
            }

        img, target = self.transforms(img, target)

        targets = {'task': self.task}
        if self.task == 'bbox':
            query = 'bound ' + target['query'] + ' with box.'
            bbox = target['boxes'][0]
            if 'refclef' in self.dataset_name:
                target_seq = bbox_process(bbox, self.num_bins, sample['cat'], 'refclef')
            elif 'refcoco' in self.dataset_name:
                target_seq = bbox_process(bbox, self.num_bins, sample['cat'], 'refcoco')
            targets.update({
                'bbox': bbox,
                'answer': target_seq
            })
        elif self.task == 'dense':
            assert self.subset != 'train', "online only support non-train dataset"
            query = 'segment ' + target['query'] + ' with mask.'
            mask = target['masks']
            if 'refclef' in self.dataset_name:
                target_seq, mask, crop_flag = dense_process(mask, None, sample['cat'], 'refclef', noseq=True)
            elif 'refcoco' in self.dataset_name:
                target_seq, mask, crop_flag = dense_process(mask, None, sample['cat'], 'refcoco', noseq=True)
            targets.update({
                'mask': mask,
                'answer': target_seq
            })
            img = img_refine_square(img, crop_flag)
        else:
            raise NotImplementedError

        return img, query, targets
    
    def get_offline(self, sample, i):
        img = self.read_image(sample['img_name'])
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img, _ = normalize(img, None)
        img = T.F.resize(img, 800)

        query = sample['sentences'][0]['sent']
        query = 'segment ' + query + ' with mask.'

        mask = self.get_mask(sample['segment_id'])
        mask = T.F.resize(mask, 256)

        fname = self.get_encoding_fname(i)
        fpath = os.path.join(self.offline_root, fname)
        file = torch.load(fpath)
        crop_flag = file['crop_flag']
        target_seq = file['answer']

        img = img_refine_square(img, crop_flag)
        mask = img_refine_square(mask, crop_flag)
        targets = {
            'task': 'dense',
            'mask': mask,
            'answer': target_seq
        }

        return img, query, targets, fname
    
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
