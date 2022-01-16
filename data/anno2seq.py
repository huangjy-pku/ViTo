import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import cv2
import yaml


def bbox_process(bbox, H, W, num_bins):
    """
    from absolute coordinates to position bins token
    input bbox: [x_min, y_min, x_max, y_max]
    return as tokens
    """
    x1, y1, x2, y2 = bbox
    token_x1 = np.clip(round(float(x1)/W*num_bins-0.5), 0, num_bins-1)
    token_y1 = np.clip(round(float(y1)/H*num_bins-0.5), 0, num_bins-1)
    token_x2 = np.clip(round(float(x2)/W*num_bins-0.5), 0, num_bins-1)
    token_y2 = np.clip(round(float(y2)/H*num_bins-0.5), 0, num_bins-1)
    return f'__bbox_begin__ pos_{str(token_x1)} pos_{str(token_y1)} pos_{str(token_x2)} pos_{str(token_y2)} __bbox_end__'


def dense_process(segment_id):
    # to be implemented
    return segment_id


def prepare_dataset(dataset, img_dir, anno_dir, output_dir, task):
    assert anno_dir != output_dir, "anno_dir should not be same as output_dir"

    for json_name in os.listdir(anno_dir):
        with open(os.path.join(anno_dir, json_name)) as f:
            src = json.load(f)

        print(f'parsing {len(src)} samples in {json_name} of {dataset}')

        dst = []
        for src_meta in tqdm(src):
            # get image shape
            img_path = os.path.join(img_dir, src_meta['img_name'])
            img = cv2.imread(img_path)
            H, W = img.shape[:-1]

            dst_meta = {
                'dataset': dataset,
                'subset': json_name.split('.')[0],
                'img_name': src_meta['img_name'],
                'img_shape': (H, W),
                'category': src_meta['cat']
            }

            if 'bbox' in task:
                # get bbox
                bbox_coordinate = src_meta['bbox']

                # get number of bins to split positions into tokens
                config_path = anno_dir.split('/')
                if '' in config_path:
                    config_path.remove('')
                config_path = os.path.join(*(config_path[:-3]), 'config', 'vito.yaml')
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                num_bins = config['model']['num_bins']

                target_seq = bbox_process(bbox_coordinate, H, W, num_bins)
                dst_meta.update({
                    'task': 'bbox',
                    'query': 'bound '+src_meta['sentences'][0]['sent']+' with box',
                    'bbox': bbox_coordinate,
                    'target_seq': target_seq,
                    })

            elif 'dense' in task:
                seg_id = src_meta['segment_id']
                target_seq = dense_process(seg_id)
                dst_meta.update({
                    'task': 'bbox',
                    'query': 'segment '+src_meta['sentences'][0]['sent']+' with mask',
                    'mask': None,
                    'target_seq': target_seq
                    })
            
            else:
                raise NotImplementedError

            dst.append(dst_meta)

        dst_dir = os.path.join(output_dir, task)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        dst_json = os.path.join(dst_dir, dataset+'_'+json_name)    
        print(f'Dumping json file to {dst_json}\n')
        with open(dst_json, 'w') as f:
            json.dump(dst, f)


def main():
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--anno_root', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'], default='refcoco')
    parser.add_argument('--task', type=str, choices=['bbox', 'dense'], default='bbox')
    args = parser.parse_args()

    if args.dataset == 'refclef':
        img_dir = os.path.join(args.data_root, 'images', 'saiapr_tc-12')
    else:
        img_dir = os.path.join(args.data_root, 'images', 'mscoco', 'images', 'train2014')

    anno_path = os.path.join(args.anno_root, args.dataset)
    
    prepare_dataset(args.dataset, img_dir, anno_path, args.output_dir, args.task)


if __name__ == '__main__':
    main()
