import numpy as np
from tqdm import tqdm
import torch

from utils.bbox_utils import compute_iou, compute_iou_mask


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
        mask_mat = np.zeros((0, 3))
        mask_iou = 0
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
                if grp['mask'].shape is not None:
                    iou, acc = self.evaluate_mask(grp['mask'], targets['mask'][0])
                    acc = acc.reshape(-1, 3)
                else:
                    iou = 0
                    acc = np.zeros((1, 3), dtype=bool)
                mask_mat = np.concatenate([mask_mat, acc], axis=0)
                mask_iou += iou
            else:
                absent += 1
                continue

        bbox_total = bbox_mat.shape[0]
        mask_total = mask_mat.shape[0]
        bbox_AP50 = bbox_mAP = None
        mask_mIoU = mask_AP = None
        
        if bbox_total > 0:
            APs = np.sum(bbox_mat, axis=0) / bbox_total
            bbox_AP50 = APs[0]
            bbox_mAP = np.mean(APs)

        if mask_total > 0:
            mask_mIoU = mask_iou / mask_total
            mask_AP = np.sum(mask_mat, axis=0) / mask_total
        
        metrics = {
            'absent': absent,
            'total': [bbox_total, mask_total],
            'bbox_AP@0.5': bbox_AP50,
            'bbox_mAP': bbox_mAP,
            'mask_mIoU': mask_mIoU,
            'mask_AP': mask_AP
        }

        return metrics
    
    def evaluate_bbox(self, pred_bbox, gt_bbox):
        """
        two metrics: AP@0.5, mAP(@0.5:0.05:0.95)
        """
        iou = compute_iou(pred_bbox, gt_bbox, fmt='xyxy')
        iou_thre = np.linspace(0.5, 0.95, 10)
        acc = iou >= iou_thre
        return acc
    
    def evaluate_mask(self, pred_mask, gt_mask):
        """
        four metrics: mIoU, AP@0.5, AP@0.7, AP@0.9
        pred_mask: ndarray [256, 256], float in [0, 1]
        gt_mask: tensor [256, 256], float in {0, 1}
        """
        pred_mask = np.array(pred_mask)
        pred_mask = pred_mask >= 0.5
        # to bool {0, 1}
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy().astype(bool)
        
        iou = compute_iou_mask(pred_mask, gt_mask)
        iou_thre = np.array([0.5, 0.7, 0.9])
        acc = iou >= iou_thre
        return iou, acc
