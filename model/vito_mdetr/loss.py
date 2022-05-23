from einops import rearrange
import torch
import torch.nn as nn


class SequenceModelingLoss(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        ignore_index = -100
        if cfg.pad_idx is not None:
            print('Ignoring pad idx:', cfg.pad_idx)
            ignore_index = cfg.pad_idx
        if weight is None:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_index)

    def filter(self, outputs, targets):
        """
        pick samples defined in sequenced task
        """
        idx_filtered_targets = [(i, t) for i, t in enumerate(targets) if 'answer' in t]
        idxs, filtered_targets = zip(*idx_filtered_targets)
        idxs = list(idxs)
        filtered_logits = outputs[idxs]
        return filtered_logits, filtered_targets

    def compute_ce_loss(self, filtered_logits, filtered_targets):
        filtered_logits = filtered_logits.permute(0, 2, 1)   # [batch_size, num_vocab, num_l_tokens]
        tgts = torch.stack([t['answer_token_ids'] for t in filtered_targets])   # [batch_size, num_l_tokens]
        loss = self.ce_loss(filtered_logits, tgts)
        return loss.mean(0).sum()

    def forward(self, outputs, targets):
        filtered_logits, filtered_targets = self.filter(outputs, targets)
        loss = self.compute_ce_loss(filtered_logits, filtered_targets)
        return loss


class CosSimLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ignore_index = -100
        if cfg.pad_idx is not None:
            print('Ignoring pad idx:', cfg.pad_idx)
            self.ignore_index = cfg.pad_idx
        
        self.cossim_loss = nn.CosineEmbeddingLoss(reduction='sum')

    def filter(self, outputs, targets):
        """
        pick samples defined in sequenced task
        """
        idx_filtered_targets = [(i, t) for i, t in enumerate(targets) if 'answer' in t]
        idxs, filtered_targets = zip(*idx_filtered_targets)
        idxs = list(idxs)
        filtered_logits = outputs[idxs]
        return filtered_logits, filtered_targets

    def compute_cossim_loss(self, filtered_logits, filtered_targets, embed):
        batch_size = filtered_logits.shape[0]
        filtered_logits = rearrange(filtered_logits, 'b n d -> (b n) d')   # [batch_size*num_l_tokens, embed_dim]
        tgts = torch.stack([t['answer_token_ids'] for t in filtered_targets])
        tgts = rearrange(tgts, 'b n -> (b n)')   # [batch_size*num_l_tokens]
        valid_idx = ~(tgts == self.ignore_index)
        tgts_embed = embed[tgts[valid_idx]]
        loss_label = torch.ones(tgts_embed.shape[0], dtype=int).to(tgts_embed.device)
        loss = self.cossim_loss(filtered_logits[valid_idx], tgts_embed, loss_label)
        return loss / batch_size

    def forward(self, outputs, targets, embed):
        filtered_logits, filtered_targets = self.filter(outputs, targets)
        loss = self.compute_cossim_loss(filtered_logits, filtered_targets, embed)
        return loss
