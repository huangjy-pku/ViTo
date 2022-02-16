import math
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import os

from .backbone import build_backbone
from .roberta import RoBERTa
from .answer_head import build_answer_head
from .loss import SequenceModelingLoss
from utils.misc import nested_tensor_from_tensor_list
from .deformable_encoder import build_deforamble_encoder


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def build_transformer_decoder(cfg):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)

    return nn.TransformerDecoder(decoder_layer, cfg.num_layers)


class AnswerInputEmbedding(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, token_ids, joint_embed):
        """
        token_ids: [batch_size, num_l_tokens]
        joint_embed: [num_vocab, roberta_dim]
        """
        bs, Tl = token_ids.shape
        embed = torch.zeros(bs, Tl, joint_embed.shape[1]).to(torch.float).to(token_ids.device)
        embed[:, :] = joint_embed[token_ids[:, :]]
        return self.transform(embed)


class ViTo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.init_params = []

        self.roberta = RoBERTa()
        self.l2v_proj = nn.Linear(cfg.roberta_dim, cfg.encoder.hidden_dim)

        self.backbone = build_backbone(cfg)
        self.encoder = build_deforamble_encoder(cfg.encoder)
        self.decoder = build_transformer_decoder(cfg.decoder)

        self.input_proj = self.build_joiner(cfg)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        answer_out_transform = nn.Linear(
            cfg.decoder.hidden_dim,
            cfg.roberta_dim)
        self.answer_head = build_answer_head(cfg, answer_out_transform)

        if 'dense' in cfg.task:
            self.code_proj = nn.Linear(cfg.code_dim, cfg.roberta_dim)
            self.vocab_expansion(cfg, cfg.vqgan_embed)
        else:
            self.vocab_expansion(cfg)

        answer_input_transform = nn.Linear(
            cfg.roberta_dim,
            cfg.decoder.hidden_dim)
        self.answer_input_embeddings = AnswerInputEmbedding(answer_input_transform)

        self.criterion = SequenceModelingLoss(self.cfg.loss)

        self.pos_enc = nn.Parameter(positionalencoding1d(
            cfg.decoder.hidden_dim, cfg.out_max_pos_len))   # for sequence in decoder
        self.pos_enc.requires_grad = False
        
        # self.task_pos_enc = torch.load(os.path.join(self.store_path, 'positional_embedding.pt'))
        # # positional embedding of roberta is of shape [512, 768]
        
        self.task_pos_enc = nn.Parameter(positionalencoding1d(
            cfg.encoder.hidden_dim, cfg.in_max_pos_len))   # for sequence in encoder
        self.task_pos_enc.requires_grad = False

        self.store_path = cfg.store_path
        # self.roberta_dict = {}

    def vocab_expansion(self, cfg, vqgan_embed_path=None):
        init_embed = 0
        if 'bbox' in cfg.task:
            """
            __bbox_begin__, __bbox_end__, num_bins x pos_token
            """
            num_bins = cfg.num_bins
            init_embed = init_embed + num_bins + 2
            self.answer_head.vocab.extend(['__bbox_begin__', '__bbox_end__'])
            self.answer_head.vocab.extend(['pos_'+str(i) for i in range(num_bins)])
        if 'dense' in cfg.task:
            init_embed += 2   # codebook inherits vqgan
            self.answer_head.vocab.extend(['__dense_begin__', '__dense_end__'])
            self.answer_head.vocab.extend(['code_'+str(i) for i in range(cfg.codebook_size)])
            assert vqgan_embed_path is not None, "dense prediction requires vqgan embedding"
            if os.path.exists(vqgan_embed_path):
                self.code_embed = torch.load(vqgan_embed_path, map_location='cpu')
                print(f'load codebook from {vqgan_embed_path}')
            else:
                self.code_embed = 0.1 * torch.randn([cfg.codebook_size, cfg.code_dim])
            self.code_embed = nn.Parameter(self.code_embed, requires_grad=False)

        if cfg.answer_head == 'linear':
            # update classifier
            self.answer_head.classifier = nn.Linear(cfg.decoder.hidden_dim, len(self.answer_head.vocab))
        
        self.repre_embed = 0.1 * torch.randn([init_embed, cfg.roberta_dim])
        self.repre_embed = nn.Parameter(self.repre_embed, requires_grad=True)

        self.vocab = self.answer_head.vocab
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

    def build_joiner(self, cfg):
        joiner_list = []
        for in_channels in self.backbone.num_channels:
            joiner_list.append(nn.Sequential(
                nn.Conv2d(in_channels, cfg.encoder.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, cfg.encoder.hidden_dim)
            ))
        if cfg.extra_conv:
            joiner_list.append(nn.Sequential(
                nn.Conv2d(in_channels, cfg.encoder.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, cfg.encoder.hidden_dim)
            ))
        return nn.ModuleList(joiner_list)

    def load_pretr_weights(self):
        """
        ViTo: [backbone, roberta.model, l2v_proj
                encoder.encoder, decoder], with common prefix module. after distributed
        MDETR: [backbone, transformer.text_encoder, transformer.resizer,
                transformer.encoder, transformer.decoder]
        """
        loaded_model = torch.load(self.cfg.pretr_weights)['model']
        curr_model = self.state_dict()
        for lk in loaded_model.keys():
            # cnn_backbone
            if 'backbone' in lk:
                if curr_model[lk].size() == loaded_model[lk].size():
                    self.init_params.append(lk)
                    curr_model[lk] = loaded_model[lk]
                    print(f'{lk} loaded')
                else:
                    print(f'{lk} size does not match')
            # roberta
            elif 'text_encoder' in lk:
                share_name = '.'.join(lk.split('.')[2:])
                roberta_lk = 'roberta.model.' + share_name
                if roberta_lk in curr_model:
                    if curr_model[roberta_lk].size() == loaded_model[lk].size():
                        self.init_params.append(roberta_lk)
                        curr_model[roberta_lk] = loaded_model[lk]
                        print(f'{roberta_lk} loaded')
                    else:
                        print(f'{roberta_lk} size does not match')
            # proj l to v
            elif 'resizer.fc' in lk:
                if 'weight' in lk:
                    l2v_lk = 'l2v_proj.weight'
                elif 'bias' in lk:
                    l2v_lk = 'l2v_proj.bias'
                else:
                    raise Exception('unknown parameter')
                if l2v_lk in curr_model:
                    if curr_model[l2v_lk].size() == loaded_model[lk].size():
                        self.init_params.append(l2v_lk)
                        curr_model[l2v_lk] = loaded_model[lk]
                        print(f'{l2v_lk} loaded')
                    else:
                        print(f'{l2v_lk} size does not match')

        self.load_state_dict(curr_model)

    def forward(self, images, queries, answer_token_ids, targets=None, fnames=None):
        task_encoding, task_mask, task_pos_enc = self.get_task_encoding(queries, fnames)
        """
        task_encoding: [batch_size, num_l_tokens, roberta_dim]
        task_mask: [batch_size, num_l_tokens]
        task_pos_enc: [num_l_tokens, hidden_dim]
        """
        task_encoding = self.l2v_proj(task_encoding)   # [batch_size, num_l_tokens, hidden_dim]
        
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        memory = self.encoder(srcs, masks, pos, task_encoding, task_mask, task_pos_enc)
        # [batch_size, num_v_tokens+num_l_tokens, hidden_dim]

        if 'dense' in self.cfg.task:
            self.joint_embed = torch.cat(
                [self.answer_head.fixed_embed, self.repre_embed.to(self.device),
                 self.code_proj(self.code_embed.to(self.device))], dim=0)
        else:
            self.joint_embed = torch.cat([self.answer_head.fixed_embed, self.repre_embed.to(self.device)], dim=0)
        # update joint_embed per forwarding

        output_logits = self.decode(answer_token_ids, memory)
        # [batch_size, num_l_tokens, num_vocab]

        if targets is None:
            # without ground truth
            return output_logits
        else:
            # with ground truth
            loss = self.criterion(output_logits, targets)
            return loss

    def check_encodings(self, fnames):
        # determine whether needed to pass forward roberta
        for fname in fnames:
            if not os.path.exists(os.path.join(self.store_path, fname)):
                return False
        return True

    def get_task_encoding(self, queries, fnames=None):
        device = self.roberta.model.device
        if fnames is None:
            with torch.no_grad():
                query_encodings, token_inputs = self.roberta(queries, device)
            mask = ~token_inputs['attention_mask'].to(torch.bool)
            pos_enc = self.task_pos_enc[:mask.shape[1]]
        else:
            flag = self.check_encodings(fnames)
            if flag:
                encoding_list = []
                length_list = []
                for fname in fnames:
                    # if fname in self.roberta_dict:
                    #     meta_dict = self.roberta_dict[fname]
                    # else:
                    encoding_path = os.path.join(self.store_path, fname)
                    meta_dict = torch.load(encoding_path, map_location='cpu')
                        # self.roberta_dict.update({fname: meta_dict})
                    encoding_list.append(meta_dict['encoding'])
                    length_list.append(meta_dict['valid_len'])
                N = len(fnames)
                max_length = max(length_list)
                query_encodings = torch.zeros(N, max_length, self.cfg.roberta_dim, device=self.device)
                mask = torch.ones(N, max_length, device=self.device).to(torch.bool)
                for i, encoding in enumerate(encoding_list):
                    query_encodings[i, :length_list[i]] = encoding
                    mask[i, :length_list[i]] = False
                
                pos_enc = self.task_pos_enc[:max_length]   # [max_length, hidden_dim]
            else:
                with torch.no_grad():
                    query_encodings, token_inputs = self.roberta(queries, device)
                valid_len = torch.count_nonzero(token_inputs['attention_mask'], dim=1)
                mask = ~token_inputs['attention_mask'].to(torch.bool)
                pos_enc = self.task_pos_enc[:mask.shape[1]]
                # save
                for i, fname in enumerate(fnames):
                    encoding_path = os.path.join(self.store_path, fname)
                    if os.path.exists(encoding_path):
                        continue
                    else:
                        meta_dict = {
                            'encoding': query_encodings[i, :valid_len[i]],
                            'valid_len': valid_len[i]
                        }
                        # self.roberta_dict.update({fname: meta_dict})
                        torch.save(meta_dict, encoding_path)
        
        return query_encodings, mask, pos_enc
    
    def decode(self, answer_token_ids, memory):
        if answer_token_ids is None:
            # inference
            B = memory.shape[0]
            cls_token_id = torch.LongTensor([self.word_to_idx['__cls__']]).cuda(self.device)
            target_token_ids = cls_token_id.view(1, 1).repeat(B, 1)
            terminate = torch.zeros(B).to(torch.bool).cuda(self.device)
            stop_token_id = torch.LongTensor([self.word_to_idx['__stop__']]).cuda(self.device)
            for t in range(self.cfg.decoder.max_answer_len-1):
                """
                auto-regressive has to get the exact word before recurrent prediction
                so the process appears like logits(t) -> token_id(t) -> logits(t+1)
                """
                target = self.answer_input_embeddings(target_token_ids, self.joint_embed)
                answer_logits = self.decode_text(target, memory)
                # [batch_size, num_predicted_tokens(t+1), num_vocab]
                answer_logits = answer_logits[:, -1]
                # current token prediction
                top_ids = torch.topk(answer_logits, k=1, dim=-1).indices

                terminate[top_ids.squeeze(-1)==stop_token_id] = True
                # early stop
                if torch.all(terminate):
                    break
                else:
                    target_token_ids = torch.cat((target_token_ids, top_ids), -1)      
            # now we have token_ids as output, which are supposed to be converted back to logits
            target = self.answer_input_embeddings(target_token_ids, self.joint_embed)
            # [batch_size, num_l_tokens, hidden_dim]
            
            output_logits = self.decode_text(target, memory)
        else:
            # train
            target = self.answer_input_embeddings(answer_token_ids[:, :-1], self.joint_embed)
            # [batch_size, num_l_tokens, hidden_dim]
            # ignore output from __stop__ token
            output_logits = self.decode_text(target, memory)
        
        return output_logits

    def encode_answers(self, targets, device=None):
        """
        transfer ground truth answers from words(list) to token_ids
        """
        self.device = device
        B = len(targets)
        answers = [''] * B
        for i, t in enumerate(targets):
            if 'answer' in t:
                answers[i] = t['answer']

        padded_inputs = [None] * len(answers)
        S = 0
        for i, answer in enumerate(answers):
            if answer == '':
                sent = f'__cls__ __stop__'
            else:
                sent = f'__cls__ {answer} __stop__'
            padded_inputs[i] = [w.lower() for w in word_tokenize(sent)]
            S = max(S, len(padded_inputs[i]))

        padded_token_ids = [None] * len(answers)
        for i, padded_tokens in enumerate(padded_inputs):
            padded_tokens.extend(['__pad__'] * (S - len(padded_tokens)))
            token_ids = [None] * S
            for j in range(S):
                if padded_tokens[j] in self.word_to_idx:
                    token_ids[j] = self.word_to_idx[padded_tokens[j]]
                else:
                    token_ids[j] = self.word_to_idx['__unk__']

            padded_token_ids[i] = token_ids[:self.cfg.decoder.max_answer_len]

        padded_token_ids = torch.LongTensor(padded_token_ids).cuda(device)

        return padded_inputs, padded_token_ids

    def token_ids_to_words(self, token_ids):
        B, S = token_ids.shape
        words = [None] * B
        for i in range(B):
            words[i] = [None] * S
            for j in range(S):
                words[i][j] = self.vocab[token_ids[i, j]]

        return words

    @property
    def cls_token(self):
        return self.answer_input_embedings(
            torch.LongTensor([self.word_to_idx['__cls__']]).cuda(self.device))[0]

    def decode_text(self, target, memory):
        Tt = target.shape[1]
        if self.cfg.decoder.pos_enc is True:
            target = target + self.pos_enc[:, :Tt]
        memory = memory.permute(1, 0, 2)   # [num_tokens, batch_size, hidden_dim]
        target = target.permute(1, 0, 2)   # [num_l_tokens, batch_size, hidden_dim]
        tgt_mask = torch.zeros((Tt, Tt)).bool().cuda(self.device)
        for t in range(Tt):
            for j in range(t+1, Tt):
                tgt_mask[t, j] = True   # True indicates not attending

        to_decode = self.decoder(target, memory, tgt_mask).permute(1, 0, 2)
        # [batch_size, num_l_tokens, hidden_dim]

        return self.answer_head(to_decode, self.joint_embed)   # [batch_size, num_l_tokens, num_vocab]
