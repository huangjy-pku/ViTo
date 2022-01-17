from operator import gt
import numpy as np
from tqdm import tqdm

from utils.bbox_utils import compute_iou


def fname_to_index(fname):
    index = fname.split('_')[-1]
    index = int(index.split('.')[0])
    return index


class RefexpEvaluator():
    def __init__(self, dataloader, pred_h5py):
        self.dataset = dataloader.dataset
        self.preds = pred_h5py

    def evaluate(self):
        absent = 0
        bbox_mat = np.zeros((0, 10))
        mask_mat = np.zeros((0, 10))
        for fname in tqdm(self.preds):
            index = fname_to_index(fname)
            img, query, targets, fname_ = self.dataset[index]
            if fname != fname_:
                absent += 1
                continue

            grp = self.preds[fname]
            if 'bbox' in grp:
                if grp['bbox'].shape is not None:
                    acc = self.evaluate_bbox(grp['bbox'],
                        targets['bbox'].detach().cpu().numpy()).reshape(-1, 10)
                else:
                    acc = np.zeros((1, 10), dtype=bool)
                bbox_mat = np.concatenate([bbox_mat, acc], axis=0)
            elif 'mask' in grp:
                acc = self.evaluate_mask(grp['mask'], targets['mask'])
                mask_mat = np.append(mask_mat, acc, axis=0)
            else:
                absent += 1
                continue

        bbox_total = bbox_mat.shape[0]
        mask_total = mask_mat.shape[0]
        
        if bbox_total > 0:
            APs = np.sum(bbox_mat, axis=0) / bbox_total
            mAP = np.mean(APs)
        else:
            APs = np.sum(mask_mat, axis=0) / mask_total
            mAP = np.mean(APs)
        
        metrics = {
            'absent': absent,
            'total': bbox_total if bbox_total > 0 else mask_total,
            'AP50': APs[0],
            'mAP': mAP
        }

        return metrics
    
    def evaluate_bbox(self, pred_bbox, gt_bbox):
        iou = compute_iou(pred_bbox, gt_bbox, fmt='xyxy')
        iou_thre = np.linspace(0.5, 0.95, 10)
        acc = iou > iou_thre
        return acc
    
    def evaluate_mask(self, pred_mask, gt_mask):
        pass
