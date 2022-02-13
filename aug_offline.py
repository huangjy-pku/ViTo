"""
Script used to generate offline augmentation samples for training of dense task.
Images and queries will be saved in 20 separate bundles.
Target sequences are supposed to be saved, so as to skip VQGAN online encoding part.
Note: this script should be run after VQGAN is prepared.
"""
import os
import hydra
from tqdm import tqdm
import torch
from dataset.generic_dataset import dense_process, GenericDataset
from taming.vqgan import VQModel


def make_aug_offline(dataset, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(len(dataset))):
        sample = dataset.samples[i]
        # img0 = dataset.read_image(sample['img_name'])
        # query0 = sample['sentences'][0]['sent']
        # bbox0 = torch.as_tensor(sample['bbox'], dtype=torch.float32).reshape(-1, 4)
        mask0 = dataset.get_mask(sample['segment_id'])
        # target0 = {
        #     'query': query0,
        #     'boxes': bbox0,
        #     'masks': mask0
        # }
        # img, target = dataset.transforms(img0, target0)
        # targets = {'task': dataset.task}
        # query = 'segment ' + target['query'] + ' with mask.'
        # mask = target['masks']
        if 'refclef' in dataset.dataset_name:
            target_seq, mask, crop_flag = dense_process(mask0, dataset.vqgan, sample['cat'], 'refclef')
        elif 'refcoco' in dataset.dataset_name:
            target_seq, mask, crop_flag = dense_process(mask0, dataset.vqgan, sample['cat'], 'refcoco')
        # img = img_refine_square(img, crop_flag)
        store_dict = {
            'crop_flag': crop_flag,
            'answer': target_seq
        }
        fname = dataset.get_encoding_fname(i)
        dst = os.path.join(dst_dir, fname)
        # print(f'Dumping json file to {dst_json}\n')
        torch.save(store_dict, dst)


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    offline_root = cfg.offline_root
    
    vqgan = VQModel(ddconfig=cfg.vqgan.ddconfig, n_embed=cfg.vqgan.n_embed,
                    embed_dim=cfg.vqgan.embed_dim, ckpt_path=cfg.vqgan.ckpt)
    vqgan.to(cfg.vqgan.device)
    vqgan.eval()

    datasets = ['refclef', 'refcoco', 'refcoco+', 'refcocog']

    for data_type in datasets:
        info = cfg.dataset[data_type]
        dataset = GenericDataset(f'{data_type}_dense', info, 'train', 'dense',
                                 num_bins=200, vqgan=vqgan, aug='online')
        make_aug_offline(dataset, offline_root)


if __name__ == '__main__':
    main()
