"""
The model in this module is used as stage-two transformer conditioning on pretrained vqgans of three modalities.
They are: image, text, target vqgan processing natural RGB image, natural language and task representation respectively.
In this way, the model is supposed to learn the joint distribution of three domains.
Or more specifically, learn tokens representing tasks conditioning on given image and language instruction.
The three modalities are concatenated in the order of image, language, target.
Note: under current setting, all encoded tokens (masks, images, texts) are passed as external arguments, only decoding modules are learnable.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .roberta import RoBERTa
from .mingpt import GPT
from taming.vqgan import VQModel


def build_image_vqgan(cfg):
    vqgan = VQModel(ddconfig=cfg.image_vqgan.ddconfig, n_embed=cfg.image_vqgan.n_embed,
                    embed_dim=cfg.image_vqgan.embed_dim, ckpt_path=cfg.image_vqgan.ckpt)
    vqgan.eval()
    return vqgan


def build_target_vqgan(cfg):
    vqgan = VQModel(ddconfig=cfg.target_vqgan.ddconfig, n_embed=cfg.target_vqgan.n_embed,
                    embed_dim=cfg.target_vqgan.embed_dim, ckpt_path=cfg.target_vqgan.ckpt)
    vqgan.eval()
    return vqgan


def build_transformer_decoder(cfg):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        nhead=cfg.nheads,
        dropout=cfg.dropout,
        activation='gelu'
)
    return nn.TransformerDecoder(decoder_layer, cfg.num_layers)


class ViTo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.roberta = RoBERTa()
        self.img_vqgan = build_image_vqgan(cfg)
        self.tgt_vqgan = build_target_vqgan(cfg)

        self.txt_proj = nn.Linear(cfg.text_filter.roberta_dim, cfg.decoder.n_embd)
        # text sequence to 3 tokens: task, object, context respectively
        self.task_query = nn.Embedding(3, cfg.decoder.n_embd)
        self.txt_filter = build_transformer_decoder(cfg.text_filter)

        self.vocab_expansion()

        self.decoder = GPT(vocab_size=self.vocab_size,
                           block_size=cfg.decoder.block_size,
                           n_layer=cfg.decoder.n_layer,
                           n_head=cfg.decoder.n_head,
                           n_embd=cfg.decoder.n_embd)
        
        self.ce_weights, enhance_dict = self.enhance_task()
        print(f'initialize loss weights with following token enhanced: {enhance_dict}')

        self.buffer_path = {'img': cfg.img_buffer,
                            'txt': cfg.txt_buffer,
                            'tgt': cfg.tgt_buffer}

    def vocab_expansion(self):
        self.vocab = ['img_'+str(i) for i in range(self.cfg.image_vqgan.n_embed)]
        self.vocab_size = self.cfg.image_vqgan.n_embed
        print(f'expand {self.cfg.image_vqgan.n_embed} image codes')
        bbox_added = False
        dense_added = False
        for task in self.cfg.task:
            self.vocab.append(f'__{task}__')
            self.vocab_size += 1
            print(f'expand {task} beginning code')
            if task == 'bbox' and not bbox_added:
                self.vocab.extend(['pos_'+str(i) for i in range(self.cfg.num_bins)])
                self.vocab_size += self.cfg.num_bins
                print(f'expand {self.cfg.num_bins} bbox codes')
                bbox_added = True
            elif not dense_added:
                self.vocab.extend(['dense_'+str(i) for i in range(self.cfg.target_vqgan.n_embed)])
                self.vocab_size += self.cfg.target_vqgan.n_embed
                print(f'expand {self.cfg.target_vqgan.n_embed} dense codes')
                dense_added = True

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

    def enhance_task(self):
        # assign higher weight for the first token in tgt_seq
        weights = torch.ones(self.vocab_size)
        enhance_dict = {}
        for w, i in self.word_to_idx.items():
            if w.startswith('__'):
                weights[i] = 100
                enhance_dict.update({i: w})
        return weights, enhance_dict

    def forward(self, buffer_names, seq_tuple, train=True):
        """
        intermediate tensors are required to be inferred and stored in buffer_path offline
        buffer_names: stacked string-ids of data samples
        """
        self.device = next(self.parameters()).device

        if seq_tuple is None:
            txt_seq, txt_pad_mask, img_seq, tgt_seq = self.get_offline(buffer_names)
        else:
            txt_seq, txt_pad_mask, img_seq, tgt_seq = self.get_online(seq_tuple)
        
        # txt_seq: [batch_size, num_txt_tokens, roberta_dim]
        # txt_pad_mask: [batch_size, num_txt_tokens]
        # img_seq: [batch_size, 256]
        # tgt_seq: [batch_size, 1+256]

        txt_seq = self.txt_proj(txt_seq).transpose(0, 1)
        task_seq = self.txt_filter(
            tgt=self.task_query.weight.unsqueeze(1).repeat(1, txt_seq.shape[1], 1),
            memory=txt_seq, memory_key_padding_mask=txt_pad_mask
        )
        task_seq = task_seq.transpose(0, 1)
        # task_seq: [batch_size, 3, hidden_dim]

        if train:
            seq_idx = torch.cat([img_seq, tgt_seq], dim=1)
            logits, _ = self.decoder(idx=seq_idx[:, :-1], embeddings=task_seq)
            # supervise target part
            logits = logits[:, -tgt_seq.shape[1]:]
            # [batch_size, 257, vocab_size]
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1),
                                   weight=self.ce_weights.to(self.device))
        else:
            seq_idx = img_seq
            for i in range(tgt_seq.shape[1]):
                logits, _ = self.decoder(idx=seq_idx, embeddings=task_seq)
                logit = logits[:, -1, :]
                _, idx = torch.topk(logit, k=1, dim=-1)
                seq_idx = torch.cat([seq_idx, idx], dim=-1)
            logits = logits[:, -tgt_seq.shape[1]:]
            return logits

    def get_offline(self, buffer_names):
        """
        return img_seqs [batch_size, len_img] with each element representing an index of image code,
        txt_seqs [batch_size, len_txt, dim_txt],
        tgt_seqs [batch_size, len_tgt] with each element representing an index of target code.
        """
        B = len(buffer_names)
        img_seqs = torch.zeros(
            (0, int((256/self.cfg.image_vqgan.downsample_factor)**2)),
            dtype=int
        )
        txt_seq_list = []
        txt_len_list = []
        tgt_seqs = torch.zeros(
            (0, int((256/self.cfg.image_vqgan.downsample_factor)**2)+1),
            dtype=int
        )
        for i in range(B):
            buffer_name = buffer_names[i]
            img_seq = torch.load(os.path.join(self.buffer_path['img'], buffer_name), map_location='cpu')
            # string to index
            img_seq = torch.tensor([self.word_to_idx[str_token] for str_token in img_seq], dtype=int)
            img_seqs = torch.cat([img_seqs, img_seq[None, :]], dim=0)

            txt_seq = torch.load(os.path.join(self.buffer_path['txt'], buffer_name), map_location='cpu')
            txt_seq_list.append(txt_seq)
            txt_len_list.append(txt_seq.shape[0])

            tgt_seq = torch.load(os.path.join(self.buffer_path['tgt'], buffer_name), map_location='cpu')
            # string to index
            tgt_seq = torch.tensor([self.word_to_idx[str_token] for str_token in tgt_seq], dtype=int)
            tgt_seqs = torch.cat([tgt_seqs, tgt_seq[None, :]], dim=0)

        txt_len_max = max(txt_len_list)
        txt_seqs = torch.zeros(B, txt_len_max, self.cfg.text_filter.roberta_dim)
        txt_pad_masks = torch.ones(B, txt_len_max).to(torch.bool)
        for i, txt_seq in enumerate(txt_seq_list):
            txt_seqs[i, :txt_len_list[i]] = txt_seq
            txt_pad_masks[i, :txt_len_list[i]] = False

        return txt_seqs.to(self.device), txt_pad_masks.to(self.device), \
               img_seqs.to(self.device), tgt_seqs.to(self.device)

    def get_online(self, seq_tuple):
        txt_seq, txt_pad_mask, img_seq, tgt_seq = seq_tuple
        img_seq_new = torch.zeros(
            (0, int((256/self.cfg.image_vqgan.downsample_factor)**2)),
            dtype=int
        )
        tgt_seq_new = torch.zeros(
            (0, int((256/self.cfg.image_vqgan.downsample_factor)**2)+1),
            dtype=int
        )
        for seq in img_seq:
            seq_new = torch.tensor([self.word_to_idx[str_token] for str_token in seq], dtype=int)
            img_seq_new = torch.cat([img_seq_new, seq_new[None, :]], dim=0)
        for seq in tgt_seq:
            seq_new = torch.tensor([self.word_to_idx[str_token] for str_token in seq], dtype=int)
            tgt_seq_new = torch.cat([tgt_seq_new, seq_new[None, :]], dim=0)
        return txt_seq.to(self.device), txt_pad_mask.to(self.device), \
               img_seq_new.to(self.device), tgt_seq_new.to(self.device)

    def token_ids_to_words(self, token_ids):
        B, S = token_ids.shape
        words = [None] * B
        for i in range(B):
            words[i] = [None] * S
            for j in range(S):
                words[i][j] = self.vocab[token_ids[i, j]]

        return words


@torch.no_grad()
def encode_txt(roberta, txts, device):
    txt_seqs, token_inputs = roberta(txts, device=device)
    txt_pad_masks = ~token_inputs['attention_mask'].to(torch.bool)
    return txt_seqs, txt_pad_masks

@torch.no_grad()
def encode_img(img_vqgan, imgs, device):
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.stack(imgs)
    
    img_input = 2*imgs - 1
    img_input = img_input.to(device)

    encoding_indices = img_vqgan.encode(img_input)[-1][-1]
    
    img_seqs = [f'img_{elem.item()}' for elem in encoding_indices]
    img_seqs = [img_seqs[i*256:(i+1)*256] for i in range(len(imgs))]
    return img_seqs

@torch.no_grad()
def encode_tgt(tgt_vqgan, tgts, task, num_bins, device):
    if task == 'bbox':
        tgt_seqs = []
        for tgt in tgts:
            # assume already normalized to [0, 1]
            tgt_seq = [f'pos_{np.clip(round(float(p)*num_bins-0.5), 0, num_bins-1)}' for p in tgt.flatten()]
            tgt_seq = ['__bbox__'] + tgt_seq
            tgt_seqs.append(tgt_seq)
    else:
        if not isinstance(tgts, torch.Tensor):
            tgts = torch.stack(tgts)
        
        tgt_input = 2*tgts - 1
        tgt_input = tgt_input.to(device).float()

        encoding_indices = tgt_vqgan.encode(tgt_input)[-1][-1]
        
        tgt_seqs = [f'dense_{elem.item()}' for elem in encoding_indices]
        tgt_seqs = [[f'__{task}__']+tgt_seqs[i*256:(i+1)*256] for i in range(len(tgts))]
    
    return tgt_seqs
