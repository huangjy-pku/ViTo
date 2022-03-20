"""
Script used to generate offline samples for training.
Images are stored as lists, each containing 256 string-mode tokens, e.g. 'img_7'.
Queries are stored as dictionaries, containing:
    tensors following RoBERTa, at shape [valid_len, roberta_dim]
    int valid_len
Target sequences are stored as lists,
    each containing 256 string-mode tokens, e.g. 'dense_5'
    prepended by '__{task}__'
    total length at 257
Note: this script should be run after VQGAN is prepared.
"""
import os
import hydra
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from dataset.generic_dataset import GenericDataset, collate_fn
from model.roberta import RoBERTa
from taming.vqgan import VQModel


def bbox_process(bbox, num_bins):
    """
    from absolute coordinates to position bins token
    input bbox: [x_min, y_min, x_max, y_max]
    return string tokens
    """
    x1, y1, x2, y2 = bbox
    token_x1 = np.clip(round(float(x1)*num_bins-0.5), 0, num_bins-1)
    token_y1 = np.clip(round(float(y1)*num_bins-0.5), 0, num_bins-1)
    token_x2 = np.clip(round(float(x2)*num_bins-0.5), 0, num_bins-1)
    token_y2 = np.clip(round(float(y2)*num_bins-0.5), 0, num_bins-1)
    return ['__bbox__', f'pos_{str(token_x1)}', f'pos_{str(token_y1)}',
            f'pos_{str(token_x2)}', f'pos_{str(token_y2)}']


def make_offline(dataloader, task, dst_dir, mod, model, device):
    os.makedirs(dst_dir, exist_ok=True)
    model.to(device)
    model.eval()
    for data in tqdm(dataloader):
        # batch_size=1
        img, query, targets = data
        buffer_name = targets[0]['buffer_name']

        if mod == 'text':
            txt_path = os.path.join(dst_dir, buffer_name)
            if not os.path.exists(txt_path):
                with torch.no_grad():
                    encoding, token_inputs = model(query, device)
                torch.save(encoding.squeeze(0), txt_path)
        elif mod == 'image':
            img_path = os.path.join(dst_dir, buffer_name)
            if not os.path.exists(img_path):
                img = 2*img[0] - 1
                img = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    encoding_indices = model.encode(img)[-1][-1]
                img_seq = []
                for idx in encoding_indices:
                    img_seq.append(f'img_{idx.item()}')
                torch.save(img_seq, img_path)
        elif mod == 'target':
            tgt_path = os.path.join(dst_dir, buffer_name)
            if not os.path.exists(tgt_path):
                if targets[0]['task'] == 'bbox':
                    tgt_seq = bbox_process(targets[0]['target'], num_bins)
                else:
                    tgt = 2*targets[0]['target'] - 1
                    tgt = tgt.unsqueeze(0).to(device)
                    with torch.no_grad():
                        encoding_indices = model.encode(tgt)[-1][-1]
                    tgt_seq = [f'__{task}__']
                    for idx in encoding_indices:
                        tgt_seq.append(f'dense_{idx.item()}')
                torch.save(tgt_seq, tgt_path)
        else:
            raise NotImplementedError


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    dst_dirs = {
        'text': cfg.model.txt_buffer,
        'image': cfg.model.img_buffer,
        'target': cfg.model.tgt_buffer
    }
    splits = ['train', 'val']
    if cfg.gpu is None:
        device = 'cpu'
    else:
        device = f'cuda:{cfg.gpu}'
    global num_bins
    num_bins = cfg.model.num_bins

    for task in cfg.task:
        for data_type, info in cfg.dataset[task].items():
            for mod in cfg.prepare:
                if mod == 'text':
                    model = RoBERTa()
                elif mod == 'image':
                    model = VQModel(ddconfig=cfg.model.image_vqgan.ddconfig, n_embed=cfg.model.image_vqgan.n_embed,
                                    embed_dim=cfg.model.image_vqgan.embed_dim, ckpt_path=cfg.model.image_vqgan.ckpt)
                elif mod == 'target':
                    model = VQModel(ddconfig=cfg.model.target_vqgan.ddconfig, n_embed=cfg.model.target_vqgan.n_embed,
                                    embed_dim=cfg.model.target_vqgan.embed_dim, ckpt_path=cfg.model.target_vqgan.ckpt)
                else:
                    raise NotImplementedError
            
                for subset in splits:
                    dataset = GenericDataset(f'{data_type}_{task}', info, subset, task)
                    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
                    make_offline(dataloader, task, dst_dirs[mod], mod, model, device)


if __name__ == '__main__':
    main()
