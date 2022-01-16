import json
import os
import nltk
import h5py
import hydra
import torch
import torch.nn as nn
import numpy as np
import skimage.io as skio
from utils.misc import collate_fn as detr_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from model.vito import ViTo
from dataset.multitask_dataset import MultitaskDataset
from utils.bbox_utils import seq2bbox, seq2mask, vis_bbox, vis_mask
import utils.io as io
from utils.html_writer import HtmlWriter
    

def visualize(model, dataloader, cfg, step, subset):
    device = f'cuda:{cfg.gpu}'
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)
    io.mkdir_if_not_exists(cfg.ckpt_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth',
        4: 'probabilities'
    })
    count = 0
    finish_vis = False
    for data in dataloader:
        imgs, queries, targets, fnames = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k, v in t.items():
                if not isinstance(v, str):
                    t[k] = v.cuda(device)
        
        answer_tokens, answer_token_ids = model.encode_answers(targets)
        outputs_logits = model(imgs, queries, answer_token_ids=None, fnames=fnames)

        dataset_name = list(dataloader.dataset.datasets.keys())[0]
        imgs, masks = dataloader.dataset.datasets[dataset_name].get_images_from_tensor(imgs)
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        masks = masks.detach().cpu().numpy()

        # visualize predictions
        pred_prob = outputs_logits.softmax(-1)
        topk = torch.topk(pred_prob, k=1, dim=-1)
        topk_ids = topk.indices.detach().squeeze().cpu().numpy()   # [batch_size, num_l_tokens]
        topk_values = topk.values.detach().squeeze().cpu().numpy()   # [batch_size, num_l_tokens]

        pred_seqs = model.token_ids_to_words(topk_ids)

        B = len(targets)
        for i, t in enumerate(targets):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            # get valid region (ignore padded region)
            valid_h = np.sum(~masks[i, :, 0])
            valid_w = np.sum(~masks[i, 0, :])
            vis_img = imgs[i, :valid_h, :valid_w]

            if t['task'] == 'bbox':
                gt = t['bbox'].detach().cpu().numpy()
                vis_bbox(gt, vis_img, color=(0, 255, 0), modify=True, fmt='xyxy')

                bbox = seq2bbox(pred_seqs[i], num_bins=model.num_bins)
                if bbox is not None:
                    vis_bbox(bbox, vis_img, color=(0, 0, 255), modify=True, fmt='xyxy')
            elif t['task'] == 'dense':
                gt = t['mask'].detach().cpu().numpy()
                vis_mask(gt, vis_img, color=(0, 255, 0), modify=True)

                mask = seq2mask(pred_seqs[i])
                vis_mask(mask, vis_img, color=(0, 0, 255), modify=True)

            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            # html_writer.add_element({
            #     0: queries[i],
            #     1: html_writer.image_tag(vis_name),
            #     2: pred_seqs[i],
            #     3: answer_tokens[i],
            #     4: np.round(topk_values[i], 4)
            # })
            html_writer.add_element({
                0: queries[i],
                1: html_writer.image_tag(vis_name),
                2: gt,
                3: answer_tokens[i],
                4: np.round(topk_values[i], 4)
            })
        
        if finish_vis is True:
            break
        
        count += B
    
    html_writer.close()


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    if cfg.gpu is not None:
        print("Use GPU: {} for testing".format(cfg.gpu))

    print(OmegaConf.to_yaml(cfg))

    datasets = {
        'train': MultitaskDataset(cfg.dataset, 'train', cfg.task, cfg.model.num_bins),
        'val': MultitaskDataset(cfg.dataset, 'val', cfg.task, cfg.model.num_bins)
    }
    for subset, dataset in datasets.items():
        print(f'{subset} set size:', len(dataset))

    model = ViTo(cfg.model)
    model.to('cuda')

    dataloaders = {}
    for subset, dataset in datasets.items():
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=detr_collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=True
        )

    model.eval()
    with torch.no_grad():
        for subset in ['train', 'val']:
            print(f'Visualizing {subset} ...')
            visualize(model, dataloaders[subset], cfg, 0, subset)
    

if __name__=='__main__':
    main()
