import json
import os
import h5py
import hydra
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.cond_vito import ViTo
from model.metrics import refexp_metrics
from dataset.generic_dataset import GenericDataset, collate_fn
from utils.bbox_utils import seq2bbox, seq2dense
import utils.io as io


def run_eval(cfg, model, tasks):
    assert tasks is not None, "task should be specified"

    testsets = {}
    for task in tasks:
        for dataset, info in cfg.dataset[task].items():
            for json_name in os.listdir(info.anno_dir):
                if 'val' in json_name or 'test' in json_name:
                    subset = json_name.split('.')[0]
                    dataset_name = f'{dataset}_{task}'
                    testsets.update({
                        f'{dataset_name}_{subset}': \
                            GenericDataset(dataset_name, info, subset, task, cfg.training.online)})

    dataloaders = {}
    for dataset_name, dataset in testsets.items():
        dataloaders[dataset_name] = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.eval.num_workers,
            pin_memory=True,
            shuffle=False)

    
    for dataset_name, dataloader in dataloaders.items():
        print(f'Evaluating on {dataset_name}')
        
        with torch.no_grad():
            metrics = refexp_metrics(model, dataloader, cfg)

        eval_str = f'Exp: {cfg.exp_name} | Dataset: {dataset_name} | reaction rate: {round(metrics["reaction_rate"], 4)} | '

        if metrics['bbox_AP@0.5'] is not None:
            bbox_AP50 = round(metrics['bbox_AP@0.5'], 4)
            bbox_mAP = round(metrics['bbox_mAP'], 4)
            eval_str += f'bbox AP@0.5: {bbox_AP50} | bbox mAP: {bbox_mAP}'
        if metrics['mask_mIoU'] is not None:
            mask_mIoU = round(metrics['mask_mIoU'], 4)
            mask_AP = metrics['mask_AP']
            mask_AP = [round(x, 4) for x in mask_AP]
            eval_str += f'mask mIoU: {mask_mIoU} | mask AP: {mask_AP}'
        if metrics['depth_l1_error'] is not None:
            depth_l1_error = round(metrics['depth_l1_error'], 4)
            eval_str += f'depth l1 error: {depth_l1_error}'
        
        print(eval_str)


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    device = f'cuda:{cfg.gpu}'
    model = ViTo(cfg.model).cuda(device)
    model.device = device

    assert os.path.exists(cfg.eval.ckpt), "checkpoint should exist!"

    ckpt = torch.load(cfg.eval.ckpt, map_location=device)
    state_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        for l in state_dict.keys():
            if l in k and state_dict[l].size() == v.size():
                state_dict[l] = v
                print(f'loaded {k}')

    model.load_state_dict(state_dict)

    last_epoch = ckpt['epoch']
    print(f'Loading checkpoint at the end of epoch {last_epoch}')
    
    run_eval(cfg, model, cfg.task)


if __name__=='__main__':
    main()
