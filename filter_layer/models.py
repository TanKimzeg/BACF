# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import torch
import torch.nn as nn
from .modules import Encoder, LayerNorm, Linear

class FMLPRecModel(nn.Module):
    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.args = args
        # self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_embeddings = Linear(args.hidden_size, args.cuda_condition)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)

        self.apply(self.init_weights)

    def add_position_embedding(self, sequence:torch.Tensor):
        # 神经网络本身对输入序列的顺序没有概念，所以需要添加位置编码
        # 这里的sequence是一个batch的序列，shape为(batch_size, seq_len)
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence) # 生成位置编码
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long() # 去除填充部分的掩码
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long() # 屏蔽未来时间步

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 结合了填充部分的掩码和未来时间步的掩码

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask, # mask只对attention有效,对傅里叶忽略
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]   # 最后一层的输出,还是一个序列

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
