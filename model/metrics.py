import os
import h5py
from tqdm import tqdm
import numpy as np
import torch

import utils.io as io
from utils.bbox_utils import seq2bbox, seq2dense
from .evaluator import Evaluator
from .cond_vito import encode_txt, encode_img, encode_tgt


def refexp_metrics(model, dataloader, cfg):
    model.eval()
    
    total = 0
    end_eval = False
    eval_dir = os.path.join(cfg.exp_dir,'train_time_eval')
    io.mkdir_if_not_exists(eval_dir)
    pred_h5py_path = os.path.join(
        eval_dir, f'{dataloader.dataset.subset}_pred.h5py')
    pred_h5py = h5py.File(pred_h5py_path, 'w')

    for data in tqdm(dataloader):
        imgs, queries, targets = data

        if targets[0]['online']:
            txt_seq, txt_pad_mask = encode_txt(model.roberta, queries, model.device)
            img_seq = encode_img(model.img_vqgan, imgs, model.device)
            tgts = []
            for target in targets:
                tgt = target['target']
                if isinstance(tgt, (list, tuple)):
                    tgt = tgt[0]
                tgts.append(tgt)
            tgt_seq = encode_tgt(model.tgt_vqgan, tgts, targets[0]['task'], cfg.model.num_bins, model.device)
            outputs_logits = model(buffer_names=None, seq_tuple=(
                txt_seq, txt_pad_mask, img_seq, tgt_seq
            ), train=False)
        else:
            buffer_names = [target['buffer_name'] for target in targets]
            outputs_logits = model(buffer_names, seq_tuple=None, train=False)

        topk = torch.topk(outputs_logits, k=1, dim=-1)
        topk_ids = topk.indices.detach().squeeze(-1).cpu().numpy()   # [batch_size, tgt_len]
        pred_seqs = model.token_ids_to_words(topk_ids)

        B = len(targets)
        for b in range(B):
            if total >= cfg.eval.num_val_samples:
                end_eval = True
                break
            
            fname = targets[b]['buffer_name']
            grp = pred_h5py.create_group(fname)
            task = targets[b]['task']

            if task == 'bbox':
                pred_bbox = seq2bbox(pred_seqs[b], num_bins=cfg.model.num_bins)
                grp.create_dataset('bbox', dtype='f', data=pred_bbox)
            elif task == 'mask':
                pred_mask = seq2dense(pred_seqs[b], model, 'mask', down_factor=cfg.model.target_vqgan.downsample_factor)
                grp.create_dataset('mask', dtype='f', data=pred_mask)
            elif task == 'depth':
                pred_depth = seq2dense(pred_seqs[b], model, 'depth', down_factor=cfg.model.target_vqgan.downsample_factor)
                grp.create_dataset('depth', dtype='f', data=pred_depth)
            
            gt = targets[b]['target']
            if isinstance(gt, (list, tuple)):
                grp.create_dataset('gt', dtype='f', data=gt[0])
                grp.create_dataset('valid_mask', dtype='bool', data=gt[1])
            else:
                grp.create_dataset('gt', dtype='f', data=gt)
                
            total += 1
        
        if end_eval:
            break

    pred_h5py.close()

    pred_h5py = h5py.File(pred_h5py_path, 'r')
    evaluator = Evaluator(pred_h5py)
    metrics = evaluator.evaluate()
    pred_h5py.close()
    os.remove(pred_h5py_path)
    return metrics
