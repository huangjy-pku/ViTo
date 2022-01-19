import json
import os
import nltk
import h5py
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import skimage.io as skio
from utils.misc import collate_fn as detr_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from pytorch_transformers.optimization import WarmupLinearSchedule

from model.vito import ViTo
from model.metrics import refexp_metrics
from dataset.multitask_dataset import MultitaskDataset
from utils.bbox_utils import seq2bbox, seq2mask, vis_bbox, vis_mask
import utils.io as io
from utils.html_writer import HtmlWriter


def grad_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    return total_norm ** (1. / 2)
    

def visualize(model, dataloader, cfg, step, subset):
    device = f'cuda:{cfg.gpu}'
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth',
        4: 'probabilities'
    })
    count = 0
    finish_vis = False
    for data in dataloader:
        imgs, queries, targets, fnames = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k, v in t.items():
                if not isinstance(v, str):
                    t[k] = v.cuda(device)
        
        answer_tokens, answer_token_ids = model.encode_answers(targets, device)
        outputs_logits = model(imgs, queries, answer_token_ids=None, fnames=fnames)

        dataset_name = list(dataloader.dataset.datasets.keys())[0]
        imgs, masks = dataloader.dataset.datasets[dataset_name].get_images_from_tensor(imgs)
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        masks = masks.detach().cpu().numpy()

        # visualize predictions
        pred_prob = outputs_logits.softmax(-1)
        topk = torch.topk(pred_prob, k=1, dim=-1)
        topk_ids = topk.indices.detach().squeeze().cpu().numpy()   # [batch_size, num_l_tokens]
        topk_values = topk.values.detach().squeeze().cpu().numpy()   # [batch_size, num_l_tokens]

        pred_seqs = model.token_ids_to_words(topk_ids)

        B = len(targets)
        for i, t in enumerate(targets):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break

            # get valid region (ignore padded region)
            valid_h = np.sum(~masks[i, :, 0])
            valid_w = np.sum(~masks[i, 0, :])
            vis_img = imgs[i, :valid_h, :valid_w]

            if t['task'] == 'bbox':
                gt = t['bbox'].detach().cpu().numpy()
                vis_bbox(gt, vis_img, color=(0, 255, 0), modify=True, fmt='xyxy')

                bbox = seq2bbox(pred_seqs[i], num_bins=cfg.model.num_bins)
                if bbox is not None:
                    vis_bbox(bbox, vis_img, color=(0, 0, 255), modify=True, fmt='xyxy')
            elif t['task'] == 'dense':
                gt = t['mask'].detach().cpu().numpy()
                vis_mask(gt, vis_img, color=(0, 255, 0), modify=True)

                mask = seq2mask(pred_seqs[i])
                vis_mask(mask, vis_img, color=(0, 0, 255), modify=True)

            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            html_writer.add_element({
                0: queries[i],
                1: html_writer.image_tag(vis_name),
                2: pred_seqs[i],
                3: answer_tokens[i],
                4: np.round(topk_values[i], 4)
            })
        
        if finish_vis is True:
            break
        
        count += B
    
    html_writer.close()


def freeze_pretr_params(model, requires_grad=False):
    print(f'Setting requires grad to False for DETR params')
    for n, p in model.named_parameters():
        if n in model.init_params:
            p.requires_grad = requires_grad


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    
    return lrs

def train_worker(gpu, cfg):
    cfg.gpu = gpu
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if gpu == 0:
        print(OmegaConf.to_yaml(cfg))

    datasets = {
        'train': MultitaskDataset(cfg.dataset, 'train', cfg.task, cfg.model.num_bins),
        'val': MultitaskDataset(cfg.dataset, 'val', cfg.task, cfg.model.num_bins)
    }
    for subset, dataset in datasets.items():
        print(f'{subset} set size:', len(dataset))

    model = ViTo(cfg.model)
    if cfg.model.pretr_weights:
        model.load_pretr_detr()
    if cfg.training.freeze is True:
        freeze_pretr_params(model)

    if cfg.multiprocessing_distributed:
        cfg.rank = cfg.rank * cfg.ngpus_per_node + cfg.gpu

        torch.cuda.set_device(cfg.gpu)
        
        dist.init_process_group(
            backend=cfg.dist_backend, 
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank)

        model.cuda(cfg.gpu)
        init_params = model.init_params
        word_to_idx = model.word_to_idx
        encode_answers = model.encode_answers
        token_ids_to_words = model.token_ids_to_words
        vocab = model.vocab
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model.encode_answers = encode_answers
        model.word_to_idx = word_to_idx
        model.token_ids_to_words = token_ids_to_words
        model.init_params = init_params
        model.vocab = vocab
        model.device = cfg.gpu

        # create sampler for dataloader
        sampler = {'val': None}
        sampler['train'] = torch.utils.data.distributed.DistributedSampler(
            datasets['train'], shuffle=True)
    else:
        model = ViTo(cfg.model)
        model.cuda(cfg.gpu)
        sampler = {'train': None, 'val': None}

    dataloaders = {}
    for subset, dataset in datasets.items():
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=detr_collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None),
            sampler=sampler[subset])

    device = f'cuda:{cfg.gpu}'
    if gpu == 0:
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    params = {
        'cnn_backbone': [],
        'roberta': [],
        'transformer': [],
        'others': []
    }
    for n, p in model.named_parameters():
        if 'backbone' in n:
            params['cnn_backbone'].append(p)
        elif 'roberta' in n:
            params['roberta'].append(p)
        elif 'encoder' in n or 'decoder' in n:
            params['transformer'].append(p)
        else:
            params['others'].append(p)

    for k, v in params.items(): 
        print(k, len(v))

    optimizer = torch.optim.AdamW([
        {'params': params['cnn_backbone'], 'lr': cfg.training.lr_backbone},
        {'params': params['transformer']},
        {'params': params['others']}],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    step = 0
    last_epoch = -1
    model_selection_metric = 0
    best_metric = 0
    best_epoch = -1
    if os.path.exists(cfg.training.ckpt):
        loc = 'cuda:{}'.format(cfg.gpu)
        ckpt = torch.load(cfg.training.ckpt, map_location=loc)
        state_dict = model.state_dict()
        for k, v in ckpt['model'].items():
            if k in state_dict and state_dict[k].size() == v.size():
                v.requires_grad = state_dict[k].requires_grad
                state_dict[k] = v
                print(k)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer'])

        step = ckpt['step']
        last_epoch = ckpt['epoch']
        if model_selection_metric in ckpt:
            model_selection_metric = ckpt['model_selection_metric']
        else:
            model_selection_metric = 0
            
        # since a checkpoint is saved only if it is has the best metric so far
        best_metric = model_selection_metric
        best_epoch = last_epoch
        print(f'Loading checkpoint at the end of epoch {last_epoch}')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.training.lr_milestones,
        cfg.training.lr_drop,
        last_epoch=last_epoch)
    
    warmup_iters = len(dataloaders['train'])
    if cfg.training.lr_warmup is True:
        if cfg.training.lr_linear_decay:
            num_train_optimization_steps = len(dataloaders['train']) * cfg.training.num_epochs
            warmup_steps = cfg.training.lr_warmup_fraction * num_train_optimization_steps
            warmup_scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=warmup_steps,
                t_total=num_train_optimization_steps,
                last_epoch=step)
        else:
            warmup_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_iters,
                last_epoch=step)   # updated every iter not epoch
            if gpu == 0:
                print('Warmup iters:', warmup_iters)

        if os.path.exists(cfg.training.ckpt):
            warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])

    if cfg.training.lr_warmup and not cfg.training.lr_linear_decay:
        # zero grad step needed for warmup scheduler
        optimizer.zero_grad()
        optimizer.step()

    training_epochs = cfg.training.num_epochs
    if cfg.training.freeze is True:
        training_epochs = cfg.training.frozen_epochs

    launch = True
    for epoch in range(last_epoch+1, training_epochs):

        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        for it, data in enumerate(dataloaders['train']):
            imgs, queries, targets, fnames = data
            imgs = imgs.to(torch.device(gpu))
            for t in targets:
                for k, v in t.items():
                    if not isinstance(v, str):
                        t[k] = v.cuda(device)
            
            model.train()

            answer_tokens, answer_token_ids = model.encode_answers(targets, device)
            for i, t in enumerate(targets):
                t['answer_token_ids'] = answer_token_ids[i, 1:]
            
            loss = model(imgs, queries, answer_token_ids, targets, fnames)

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                if cfg.training.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params['cnn_backbone']+params['others'], 
                        cfg.training.clip_max_norm
                    )
                optimizer.step()
            
            if gpu == 0 and step % cfg.training.log_step == 0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                if cfg.training.lr_linear_decay:
                    loss_str += f' LR: {warmup_scheduler.get_last_lr()[0]} | '
                loss_value = round(loss.item(), 4)
                writer.add_scalar('Epoch', epoch, step)
                writer.add_scalar('Iter', it, step)
                writer.add_scalar('Step', step, step)
                writer.add_scalar('Best Epoch', best_epoch, step)
                for j, group_lr in enumerate(get_lrs(optimizer)):
                    writer.add_scalar(
                        f'Lr/optimizer/group_{j}',
                        group_lr,
                        step
                    )
                
                loss_value = round(loss_value, 4)
                loss_str += f'Loss: {loss_value} | '
                writer.add_scalar(f'Loss/train', loss_value, step)
                print(loss_str)

                # detail_str = 'detail: '
                # for loss_name, loss_value in loss_detail.items():
                #     if loss_value is None:
                #         continue
                #     loss_name = loss_name.replace('loss_', '')
                #     loss_value = round(loss_value.item(),4)
                #     detail_str += f'{loss_name} {loss_value} + '
                # detail_str = detail_str[:-3]
                # print(detail_str)

            if gpu == 0 and step % cfg.training.vis_step == 0 and \
                ((not launch) or cfg.training.run_vis_at_launch):
                with torch.no_grad():
                    model.eval()
                    for subset in ['train', 'val']:
                        print(f'Visualizing {subset} ...')
                        visualize(model, dataloaders[subset], cfg, step, subset)

            if gpu == 0 and step % (10*cfg.training.log_step) == 0:
                print('Exp:', cfg.exp_name)
                
            step += 1
            launch = False

            if cfg.training.lr_linear_decay:
                warmup_scheduler.step()
            elif cfg.training.lr_warmup is True and epoch == 0 and it < warmup_iters:
                warmup_scheduler.step(it)

        if not cfg.training.lr_linear_decay:
            lr_scheduler.step()
        
        if gpu == 0 and ((not launch) or cfg.training.run_eval_at_launch):
            model_selection_metric = 0
            for eval_subset in ['train', 'val']:
                for dataset_name in dataloaders[eval_subset].dataset.datasets:
                    print(f'Evaluating on {dataset_name}')
                    eval_dataset = dataloaders[eval_subset].dataset.datasets[dataset_name]
                    eval_dataloader = DataLoader(
                        eval_dataset,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.workers,
                        shuffle=True,
                        collate_fn=detr_collate_fn)
                    
                    with torch.no_grad():
                        ap50, mAP = refexp_metrics(model, eval_dataloader, cfg)
                    if eval_subset == 'val':
                        model_selection_metric = model_selection_metric + ap50 + mAP
                    ap50 = round(ap50, 4)
                    mAP = round(mAP, 4)
                    print(f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | AP@0.5: {ap50} | mAP: {mAP}')
                    
                    writer.add_scalar(f'{eval_subset}/{dataset_name}/AP@0.5', ap50, step)
                    writer.add_scalar(f'{eval_subset}/{dataset_name}/mAP', mAP, step)

            if model_selection_metric > best_metric:
                print('Saving checkpoint ...')
                best_metric = model_selection_metric
                best_epoch = epoch
                writer.add_scalar('Best Epoch', best_epoch, step)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                    'step': step,
                    'lr': lr_scheduler.get_last_lr(),
                    'model_selection_metric': model_selection_metric,
                    'warmup_scheduler': warmup_scheduler.state_dict() if cfg.training.lr_linear_decay else None,
                }, os.path.join(cfg.ckpt_dir, 'model.pth'))


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir, recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir, recursive=True)
    io.mkdir_if_not_exists(cfg.model.store_path, recursive=True)
    # nltk.download('punkt')
    
    if cfg.training.freeze:
        cfg.training.batch_size = cfg.training.frozen_batch_size
        cfg.batch_size = cfg.training.frozen_batch_size

    if cfg.multiprocessing_distributed:
        cfg.world_size = cfg.ngpus_per_node * cfg.num_nodes
        cfg.batch_size = int(cfg.batch_size / cfg.ngpus_per_node)
        cfg.workers = int(
            (cfg.workers + cfg.ngpus_per_node - 1) / cfg.ngpus_per_node
        )

        mp.spawn(train_worker, nprocs=cfg.ngpus_per_node, args=(cfg,))
    else:
        train_worker(cfg.gpu, cfg)
    

if __name__=='__main__':
    main()
