from fairseq.modules.fairseq_dropout import FairseqDropout
import fairseq
from fairseq.modules.layer_norm import LayerNorm
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from torch.functional import Tensor
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
import torch
from fairseq.modules.transformer_layer import FeedForwardNetwork, TransformerEncoderLayer
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq import utils
import torch.nn as nn
from smoe.moe_layer import MoELayer
from smoe.htop2gate import HTop2Gate

class SmoeEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(args, dictionary, embed_tokens)
        moe_freq = max(getattr(args, 'encoder_moe_freq', 0), getattr(args, 'moe_freq', 0))
        assert moe_freq==2, 'currently only supports moe-freq=2'
        self.encoder_layers=[self.build_encoder_layer(args, is_moe_layer=False) 
                        for _ in range(args.encoder_layers//2)]
        self.attn_layers=[self.build_layer(args, EncoderSelfAttnLayer) 
                        for _ in range(args.encoder_layers//2)]
        self.ffn_layers=[self.build_layer(args, FFNEncoderLayer),]
        self.moe_layers=[]
        for i in range(args.encoder_layers//2):
            layer_idx=i if i!=(args.encoder_layers//2-1) else -1
            self.moe_layers.append(self.build_layer(args, SmoeEncoderLayer, layer_idx=layer_idx))
        del self.layers
        self.layers=[]
        for i in range(len(self.encoder_layers)):
            self.layers.append(self.encoder_layers[i])
            self.layers.append(self.attn_layers[i])
            self.layers.append(self.moe_layers[i])

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def build_layer(self, args, layer_cls, **kwargs):
        layer = layer_cls(args, **kwargs)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lang_id: Optional[torch.Tensor] = None,
        src_lang_id: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # logger.warning(f'zxy check if topk status pass to moe {subset_id}')
        return self.forward_scriptable(src_tokens,
                                       src_lengths,
                                       tgt_lang_id,
                                       return_all_hiddens,
                                       token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lang_id: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())
        
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        l_aux = []
        for layer in self.layers:
            if isinstance(layer, SmoeEncoderLayer):
                x, l_aux_i = layer(
                x, 
                tgt_lang_id=tgt_lang_id, 
                encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            else:
                x, l_aux_i = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        
        if not self.training and getattr(self.args, 'record_token_expert', False):
            self.collect_expert_choices(src_tokens)
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "l_aux": l_aux,
        }

    def forward_torchscript(self, net_input: Dict[str, Tensor], tgt_lang_id=None):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input, tgt_lang_id)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor], tgt_lang_id=None):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input, tgt_lang_id=tgt_lang_id)


class EncoderSelfAttnLayer(TransformerEncoderLayer):
    def __init__(self, args) -> None:
        super().__init__(args)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)
        
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        return x, None

class FFNEncoderLayer(nn.Module):
    def __init__(self, args, ) -> None:
        super().__init__()
        self.is_moe_layer=False
        self.embed_dim = args.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        ffn_dim = args.encoder_ffn_embed_dim
        self.layer_norm=LayerNorm(self.embed_dim)
        self.ffn=FeedForwardNetwork(args, self.embed_dim, ffn_dim, self.dropout_module)
    
    def forward(self, x, encoder_padding_mask=None):
        res=x
        x=self.layer_norm(x)
        x=self.ffn(x)
        x=res+x
        return x, None

    def upgrade_state_dict_named(self, *args, **kwargs):
        pass

class SmoeEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, layer_idx):        
        self.layer_idx=layer_idx
        super().__init__(args, is_moe_layer=True)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if 'attn' in n and hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)

    def forward(self, x, tgt_lang_id: Optional[Tensor], encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        layer_idx=self.moe_layer.layer_idx
        norm_before=self.normalize_before

        res=x
        if norm_before:
            x = self.final_layer_norm(x)
        
        x = x.transpose(0, 1) # batch_size, seq_len, model_dim
        if getattr(self.args, "use_moe_pad_mask", False):
            x, l_aux = self.moe_layer(x, input_padding_mask=encoder_padding_mask, tgt_lang_id=tgt_lang_id,)
        else:
            x, l_aux = self.moe_layer(x, tgt_lang_id=tgt_lang_id)
        x = x.transpose(0, 1) # seq_len, batch_size, model_dim
        x = self.residual_connection(x, res)

        if not norm_before:
            x = self.final_layer_norm(x)
        return x, l_aux
    
    def build_moe_layer(self, gate, experts, args):
        return MoELayer(gate, experts, args, self.embed_dim, self.layer_idx, autoregressive=False)
    
    def make_experts(self, args, embed_dim, expert_ffn_dim, dropout_module):
        expert_dict={
            'global':[]
        }

        world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        ddp_rank = fairseq.distributed_utils.get_data_parallel_rank()
        start_seed = torch.randint(1000000, (1,)).item()
        
        # at least as many experts than gpus
        assert args.moe_expert_count >= world_size and args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_dict['global'].append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))

        expert_dict['global']=nn.ModuleList(expert_dict['global'])
        experts = nn.ModuleDict(expert_dict)
        return experts

    def build_gate(self, args):
        if args.hmoe_gate:
            gate = HTop2Gate(
                self.embed_dim,
                args.moe_expert_count,
                args.moe_gating_use_fp32,
                args.moe_second_expert_policy,
                args.moe_normalize_gate_prob_before_dropping,
                getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                getattr(args, "moe_batch_prioritized_routing", False),
                getattr(args, "capacity_factor", 1.0),
                getattr(args, "add_lang_loss", False),
                getattr(args, "lang_dict", None),
            )
        else:
            gate=super().build_gate(args)
        return gate

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        torch.nn.LayerNorm
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
