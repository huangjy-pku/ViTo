import json
import os
import nltk
import h5py
import hydra
import torch
import torch.nn as nn
import numpy as np
from utils.misc import collate_fn as detr_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vito import ViTo
from model.metrics import refexp_metrics
from dataset.generic_dataset import GenericDataset
from utils.bbox_utils import seq2bbox, seq2mask
import utils.io as io
from taming.vqgan import VQModel


def run_eval(cfg, vqgan):
    testset = {}
    for dataset, info in cfg.dataset.items():
        for json_name in os.listdir(info.anno_dir):
            if 'val' in json_name or 'test' in json_name:
                for task in cfg.task:
                    subset = json_name.split('.')[0]
                    dataset_name = f'{dataset}_{task}'
                    testset.update({
                        f'{dataset_name}_{subset}': \
                            GenericDataset(dataset_name, info, subset, task, cfg.model.num_bins, vqgan, cfg.training.augmentation)})

    dataloaders = {}
    for dataset_name, dataset in testset.items():
        dataloaders[dataset_name] = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            collate_fn=detr_collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=False)

    model = ViTo(cfg.model).cuda(cfg.gpu)
    model.device = cfg.gpu

    assert os.path.exists(cfg.eval.ckpt), "checkpoint should exist!"

    loc = 'cuda:{}'.format(cfg.gpu)
    ckpt = torch.load(cfg.eval.ckpt, map_location=loc)
    state_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        for l in state_dict.keys():
            if l in k and state_dict[l].size() == v.size():
                state_dict[l] = v
                print(k)

    model.load_state_dict(state_dict)

    last_epoch = ckpt['epoch']
    print(f'Loading checkpoint at the end of epoch {last_epoch}')
    
    for dataset_name, dataloader in dataloaders.items():
        print(f'Evaluating on {dataset_name}')
        
        with torch.no_grad():
            metrics = refexp_metrics(model, dataloader, cfg, vqgan)

        eval_str = f'Exp: {cfg.exp_name} | Dataset: {dataset_name} | '

        if metrics['bbox_AP@0.5'] is not None:
            bbox_AP50 = round(metrics['bbox_AP@0.5'], 4)
            bbox_mAP = round(metrics['bbox_mAP'], 4)
            eval_str += f'bbox AP@0.5: {bbox_AP50} | bbox mAP: {bbox_mAP}'
        if metrics['mask_mIoU'] is not None:
            mask_mIoU = round(metrics['mask_mIoU'], 4)
            mask_AP = metrics['mask_AP']
            mask_AP = [round(x, 4) for x in mask_AP]
            eval_str += f'mask mIoU: {mask_mIoU} | mask AP: {mask_AP}'
        
        print(eval_str)


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    global vqgan
    vqgan = None
    if 'dense' in cfg.task:
        vqgan = VQModel(ddconfig=cfg.vqgan.ddconfig, n_embed=cfg.vqgan.n_embed,
                        embed_dim=cfg.vqgan.embed_dim, ckpt_path=cfg.vqgan.ckpt)
        vqgan.to(cfg.vqgan.device)
        vqgan.eval()
    
    run_eval(cfg, vqgan)


if __name__=='__main__':
    main()
