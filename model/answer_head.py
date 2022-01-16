import torch
import torch.nn as nn
import numpy as np

import utils.io as io


class AnswerHead(nn.Module):
    def __init__(
            self,
            vocab,
            embed_dim,
            hidden_dim,
            classifier_transform,
            vocab_embed=None):
        super().__init__()
        self.vocab = vocab

        if vocab_embed is None:
            vocab_embed = 0.1 * torch.randn([len(vocab), embed_dim])
        else:
            vocab_embed = torch.FloatTensor(vocab_embed)
        
        self.fixed_embed = nn.Parameter(vocab_embed, requires_grad=False)
        self.classifier_transform = classifier_transform

    def forward(self, answer_embed, joint_embed):
        """
        answer_embed: [batch_size, num_l_tokens, hidden_dim]
        joint_embed: [num_vocab, roberta_dim]
        """
        answer_embed = self.classifier_transform(answer_embed)   # [batch_size, num_l_tokens, roberta_dim]
        # compute logits by inner product between prediction and vocab candidates
        return torch.matmul(answer_embed, joint_embed.permute(1, 0))   # [batch_size, num_l_tokens, num_vocab]


class LinearAnswerHead(nn.Module):
    def __init__(
            self,
            vocab,
            embed_dim,
            hidden_dim,
            vocab_embed=None):
        super().__init__()
        self.vocab = vocab
        
        if vocab_embed is None:
            vocab_embed = 0.1 * torch.randn([len(self.vocab), embed_dim])
        else:
            vocab_embed = torch.FloatTensor(vocab_embed)
        
        self.fixed_embed = nn.Parameter(vocab_embed, requires_grad=False)
        self.classifier = nn.Linear(hidden_dim, len(vocab))

    def forward(self, answer_embed):
        return self.classifier(answer_embed)

def build_answer_head(cfg, classifier_transform):
    vocab_embed = None
    if cfg.vocab_embed is not None:
        vocab_embed = np.load(cfg.vocab_embed)
    
    vocab = io.load_json_object(cfg.vocab)
    if cfg.answer_head == 'linear':
        return LinearAnswerHead(vocab, cfg.roberta_dim, cfg.decoder.hidden_dim, vocab_embed)

    return AnswerHead(
        vocab,
        cfg.roberta_dim,
        cfg.decoder.hidden_dim,
        classifier_transform,
        vocab_embed)
