import torch
import torch.nn as nn


class SequenceModelingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ignore_index = -100
        if cfg.pad_idx is not None:
            print('Ignoring pad idx:', cfg.pad_idx)
            ignore_index = cfg.pad_idx

        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

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
