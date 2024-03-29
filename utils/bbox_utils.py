import numpy as np
import skimage.draw as skdraw
import torch
import cv2
from torch.nn.functional import interpolate


def seq2bbox(pred_seq, num_bins=200):
    """
    pred_seq: [num_l_tokens]
    return bbox or None
    """

    if pred_seq[0] != '__bbox__':
        return None

    bbox = []
    for token in pred_seq[1:]:
        if 'pos_' in token:
            bin_id = int(token[4:])
            bbox.append((bin_id+0.5)/num_bins)
        else:
            return None
    return np.array(bbox)


def seq2dense(pred_seq, model, task, down_factor=16):
    if pred_seq[0] != f'__{task}__':
        return None
    
    side = int(256 / down_factor)
    code=[]

    for token in pred_seq[1:]:
        if 'dense_' in token:
            code_id = int(token[6:])
            code.append(code_id)
        else:
            return None
    with torch.no_grad():
        pred_dense = model.tgt_vqgan.decode_code(torch.LongTensor(code).to(model.device),
                                      shape=(1, side, side, -1))
    # vqgan reconstruction shape at [1, 3, 256, 256], value in [-1, 1]
    pred_dense = pred_dense.squeeze().detach().cpu().numpy()
    pred_dense = (pred_dense+1) / 2
    if pred_dense.ndim == 3:
        # squeeze 3 channels
        # ITU-R 601-2 luma transform: L = R * 0.299 + G * 0.587 + B * 0.114
        pred_dense = np.sum([[[0.299]], [[0.587]], [[0.114]]]*pred_dense, axis=0)
    return pred_dense


def vis_bbox(bbox, img, color=(255, 0, 0), modify=False, fmt='ncxcywh'):
    im_h, im_w = img.shape[:2]
    if fmt == 'cxcywh':
        bbox = cxcywh_to_xyxy(bbox)
    elif fmt == 'ncxcywh':
        bbox = cxcywh_to_xyxy(bbox, im_h, im_w)
    elif fmt == 'xyxy':
        pass
    else:
        raise NotImplementedError(f'fmt={fmt} not implemented')

    x1, y1, x2, y2 = bbox * [im_w, im_h, im_w, im_h]
    x1 = max(0, min(x1, im_w-1))
    x2 = max(x1, min(x2, im_w-1))
    y1 = max(0, min(y1, im_h-1))
    y2 = max(y1, min(y2, im_h-1))
    # r = [y1, y1, y2, y2]
    # c = [x1, x2, x2, x1]

    if modify == True:
        img_ = np.ascontiguousarray(img)
    else:
        img_ = np.copy(img)

    if len(img.shape) == 2:
        color = (color[0],)

    # rr, cc = skdraw.polygon_perimeter(r, c, img.shape[:2])   # curve
    
    # if len(img.shape) == 3:
    #     for k in range(3):
    #         img_[rr, cc, k] = color[k]
    # elif len(img.shape) == 2:
    #     img_[rr, cc] = color[0]
    cv2.rectangle(img_, [int(x1),int(y1)], [int(x2),int(y2)], color, 5)

    return img_


def vis_mask(mask, img, color=(255, 0, 0), modify=False, alpha=0.2):
    if modify == True:
        img_ = img
    else:
        img_ = np.copy(img)
    
    # mask shape may not match img shape
    if mask.shape != img_.shape[:2]:
        # ndarray [256, 256] -> tensor [1, 1, 256, 256] -> ndarray [256, 256]
        mask = torch.from_numpy(mask)
        mask = interpolate(mask[None, None].float(), img_.shape[:2], mode="nearest")[0, 0]
        mask = mask.numpy()
    
    if mask.dtype != np.uint8:
        mask = np.clip(255*mask, 0, 255).astype(np.uint8)
    
    rr, cc = mask.nonzero()
    skdraw.set_color(img_, (rr, cc), color, alpha=alpha)   # area
    return img_, mask


def compute_iou(bbox1, bbox2, fmt='cxcywh', verbose=False):
    if fmt in ['cxcywh' or 'ncxcywh']:
        bbox1 = cxcywh_to_xyxy(bbox1)
        bbox2 = cxcywh_to_xyxy(bbox2)
    elif fmt=='xyxy':
        pass
    else:
        raise NotImplementedError(f'fmt={fmt} not implemented')

    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    
    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1, invalid=0.0)
    area2 = compute_area(bbox2, invalid=0.0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 


def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def compute_center(bbox):
    x1,y1,x2,y2 = bbox
    xc = 0.5*(x1+x2)
    yc = 0.5*(y1+y2)
    return (xc,yc)


def cxcywh_to_xyxy(bbox,im_h=1,im_w=1):
    cx,cy,w,h = bbox
    cx,cy,w,h = cx*im_w,cy*im_h,w*im_w,h*im_h
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return (x1,y1,x2,y2)


def compute_iou_mask(pred_mask, gt_mask):
    """
    masks are both bool type
    """
    inter = np.sum(pred_mask*gt_mask)
    union = np.sum(pred_mask+gt_mask)
    iou = inter / (union+1e-6)
    return iou
