import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class RoBERTa(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, sentences, device=None):
        token_inputs = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt')
        
        if device is None:
            token_inputs = {k: v.cuda() for k, v in token_inputs.items()}
        else:
            token_inputs = {k: v.cuda(device) for k, v in token_inputs.items()}
        
        outputs = self.model(**token_inputs)
        return outputs[0], token_inputs


if __name__=='__main__':
    roberta = RoBERTa(None).to('cuda:2')
    # seq_pair = roberta.tokenizer.encode("__bbox_begin__ pos_1 pos_10 pos_20 pos_30 __bbox_end__")
    # print(roberta.tokenizer.decode(seq_pair))
    # print(roberta(['How do you do?','I am fine thank you.']))
    print(roberta([
        'locate a blue turtle-like pokemon with round head with box',
        'locate "a blue turtle-like pokemon with round head" with box',
        'locate " a blue turtle-like pokemon with round head " with box'
    ], device='cuda:2'))
