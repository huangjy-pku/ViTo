import os
import h5py
from tqdm import tqdm
import numpy as np
import torch

import utils.io as io
from utils.bbox_utils import seq2bbox, seq2mask
from .evaluator import RefexpEvaluator


def refexp_metrics(model, dataloader, cfg, vqgan):
    device = f'cuda:{cfg.gpu}'

    model.eval()
    
    total = 0
    end_eval = False
    eval_dir = os.path.join(cfg.exp_dir,'train_time_eval')
    io.mkdir_if_not_exists(eval_dir)
    pred_h5py_path = os.path.join(
        eval_dir, f'{dataloader.dataset.subset}_pred.h5py')
    pred_h5py = h5py.File(pred_h5py_path, 'w')

    for data in tqdm(dataloader):
        imgs, queries, targets, fnames = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k, v in t.items():
                if v is not None and not isinstance(v, str):
                    t[k] = v.cuda(device)

        outputs_logits = model(imgs, queries, answer_token_ids=None, fnames=fnames)

        topk = torch.topk(outputs_logits, k=1, dim=-1)
        topk_ids = topk.indices.detach().squeeze(-1).cpu().numpy()   # [batch_size, num_l_tokens]
        pred_seqs = model.token_ids_to_words(topk_ids)

        B = len(targets)
        for b in range(B):
            if total >= cfg.training.num_val_samples:
                end_eval = True
                break
            
            fname = fnames[b]
            grp = pred_h5py.create_group(fname)
            task = targets[b]['task']

            if task == 'bbox':
                pred_bbox = seq2bbox(pred_seqs[b], num_bins=cfg.model.num_bins)
                grp.create_dataset('bbox', dtype='f', data=pred_bbox)
            elif task == 'dense':
                pred_mask = seq2mask(pred_seqs[b], vqgan, down_factor=cfg.vqgan.downsample_factor)
                grp.create_dataset('mask', dtype='f', data=pred_mask)
                
            total += 1
        
        if end_eval:
            break

    pred_h5py.close()

    pred_h5py = h5py.File(pred_h5py_path, 'r')
    refexp_evaluator = RefexpEvaluator(dataloader, pred_h5py)
    metrics = refexp_evaluator.evaluate()
    pred_h5py.close()
    os.remove(pred_h5py_path)
    return metrics
