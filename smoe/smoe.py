import torch
from fairseq.models.transformer import TransformerModel
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import distributed_utils, utils
from smoe.smoe_encoder import SmoeEncoder
from smoe.smoe_decoder import SmoeDecoder
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import torch.nn as nn
import os
import logging
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, num_tasks, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.emb = nn.Embedding(num_tasks, in_features)
        with torch.no_grad():
            self.emb.weight.normal_(mean=0, std=1.0)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

@register_model("smoe")
class SMoE(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.num_tasks = args.num_tasks
        self.gate_task_specific_dim = args.task_emb_dim
        self.global_lang_list = torch.arange(self.num_tasks, dtype=torch.int)
        
        if args.hmoe_gate:
            self.task_mlp = MLP(
                num_tasks = self.num_tasks,
                in_features=self.gate_task_specific_dim, 
                hidden_features=int(self.gate_task_specific_dim), 
                out_features=self.gate_task_specific_dim,)
            
            if args.task_mlp_path and os.path.exists(args.task_mlp_path):
                logger.info('Load pretrained task mlp weights from {}'.format(args.task_mlp_path))
                pretrained_weights = torch.load(args.task_mlp_path)
                self.task_mlp.load_state_dict(pretrained_weights, strict=True)
        else:
            self.task_mlp = None
            
    @classmethod
    def build_model(cls, args, task):
        world_size=distributed_utils.get_global_world_size()
        local_world_size=torch.cuda.device_count()
        assert world_size%local_world_size==0, f'local_world_size:{local_world_size}, world_size:{world_size}'
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SmoeEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SmoeDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_lang_id,
        tgt_lang_id,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if (tgt_lang_id is not None) and (self.task_mlp is not None):
            # compute lang emb [bsz, 1] > [bsz, 1, dim]
            tgt_lang_emb = self.task_mlp(tgt_lang_id)

            if self.args.add_lang_loss and self.training:
                global_lang_list = self.global_lang_list.to(tgt_lang_id.device)
                global_lang_emb = self.task_mlp(global_lang_list)
                global_lang_emb = global_lang_emb.view(global_lang_emb.shape[0],1,global_lang_emb.shape[1])
                tgt_lang_emb = torch.cat((tgt_lang_emb,global_lang_emb),dim=0)     
        else:
            tgt_lang_emb = None
        encoder_out = self.encoder(
                src_tokens=src_tokens, 
                src_lengths=src_lengths, 
                tgt_lang_id=tgt_lang_emb, 
                return_all_hiddens=return_all_hiddens
            )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            tgt_lang_id=tgt_lang_emb, 
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hmoe-gate', action='store_true', default=False, help="")
        parser.add_argument('--add-lang-loss', action='store_true', default=False, help="")
        parser.add_argument('--num-tasks', type=int, default=101, help="number of tasks (languages, 101 for OPUS)")
        parser.add_argument('--task-emb-dim', type=int, default=512, help="dimension of task embedding")
        parser.add_argument('--task-mlp-path', type=str, default=None, help="path to pretrained task mlp weights")
        parser.add_argument('--reorder-layers', action='store_true', default=False, help="")
        parser.add_argument('--layernorm-after-moe-layer', action='store_true', default=False)
        TransformerModel.add_args(parser)        

@register_model_architecture('smoe', 'smoe')
def smoe(args):
    pass