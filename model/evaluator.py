import numpy as np
from tqdm import tqdm
import torch

from utils.bbox_utils import compute_iou, compute_iou_mask


def fname_to_index(fname):
    index = fname.split('_')[-1]
    index = int(index.split('.')[0])
    return index


class Evaluator():
    def __init__(self, h5py_file):
        self.h5py = h5py_file

    def evaluate(self):
        total = len(self.h5py)
        absent = 0
        bbox_mat = np.zeros((0, 10))
        mask_mat = np.zeros((0, 3))
        mask_iou = 0
        depth_err = []
        for fname in tqdm(self.h5py):
            grp = self.h5py[fname]
            if 'bbox' in grp:
                if grp['bbox'].shape is not None:
                    acc = self.evaluate_bbox(grp['bbox'], grp['gt']).reshape(-1, 10)
                    bbox_mat = np.concatenate([bbox_mat, acc], axis=0)
                else:
                    absent += 1
            elif 'mask' in grp:
                if grp['mask'].shape is not None:
                    iou, acc = self.evaluate_mask(grp['mask'], grp['gt'])
                    acc = acc.reshape(-1, 3)
                    mask_mat = np.concatenate([mask_mat, acc], axis=0)
                    mask_iou += iou
                else:
                    absent += 1
            elif 'depth' in grp:
                if grp['depth'].shape is not None:
                    l1_error = self.evaluate_depth(grp['depth'], grp['gt'])
                    depth_err.append(l1_error)
                else:
                    absent += 1
            else:
                absent += 1

        bbox_total = bbox_mat.shape[0]
        mask_total = mask_mat.shape[0]
        depth_total = len(depth_err)
        bbox_AP50 = bbox_mAP = None
        mask_mIoU = mask_AP = None
        depth_err_mean = None
        
        if bbox_total > 0:
            APs = np.sum(bbox_mat, axis=0) / bbox_total
            bbox_AP50 = APs[0]
            bbox_mAP = np.mean(APs)

        if mask_total > 0:
            mask_mIoU = mask_iou / mask_total
            mask_AP = np.sum(mask_mat, axis=0) / mask_total
        
        if depth_total > 0:
            depth_err_mean = np.array(depth_err).mean()
        
        metrics = {
            'reaction_rate': 1-absent/total,
            'bbox_AP@0.5': bbox_AP50,
            'bbox_mAP': bbox_mAP,
            'mask_mIoU': mask_mIoU,
            'mask_AP': mask_AP,
            'depth_l1_error': depth_err_mean
        }

        return metrics
    
    def evaluate_bbox(self, pred_bbox, gt_bbox):
        """
        two metrics: AP@0.5, mAP @(0.5:0.05:0.95)
        """
        if isinstance(gt_bbox, torch.Tensor):
            gt_bbox = gt_bbox.detach().cpu().numpy()
        
        iou = compute_iou(pred_bbox, gt_bbox, fmt='xyxy')
        iou_thre = np.linspace(0.5, 0.95, 10)
        acc = iou >= iou_thre
        return acc
    
    def evaluate_mask(self, pred_mask, gt_mask):
        """
        four metrics: mIoU, AP@0.5, AP@0.7, AP@0.9
        pred_mask: ndarray [256, 256]
        gt_mask: tensor [256, 256] or [1, 256, 256], float in {0, 1}
        """
        pred_mask = np.array(pred_mask)
        pred_mask = pred_mask >= 0.5
        # to bool {0, 1}
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy().squeeze().astype(bool)
        else:
            gt_mask = np.array(gt_mask).squeeze().astype(bool)
        
        iou = compute_iou_mask(pred_mask, gt_mask)
        iou_thre = np.array([0.5, 0.7, 0.9])
        acc = iou >= iou_thre
        return iou, acc
    
    def evaluate_depth(self, pred_depth, gt_depth):
        """
        metric: l1 error averaged on pixels
        pred_depth: ndarray [256, 256]
        gt_depth: tensor [256, 256] or [1, 256, 256], float in [0, 1]
        """
        pred_depth = np.array(pred_depth).squeeze()

        valid_mask = None
        if isinstance(gt_depth, tuple):
            gt_depth, valid_mask = gt_depth

        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.detach().cpu().numpy().squeeze()
        else:
            gt_depth = np.array(gt_depth).squeeze()
        
        if valid_mask is None:
            return np.abs(pred_depth-gt_depth).mean()
        else:
            if isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.detach().cpu().numpy().squeeze()
            else:
                valid_mask = np.array(valid_mask).squeeze()
            return np.abs(pred_depth-gt_depth).mean(where=valid_mask)
