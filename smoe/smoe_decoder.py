from fairseq.modules.fairseq_dropout import FairseqDropout
import fairseq
from fairseq.modules.layer_norm import LayerNorm
from typing import Dict, List, Optional, Any
from torch.functional import Tensor
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
import torch
from fairseq.modules.transformer_layer import FeedForwardNetwork, TransformerDecoderLayer
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerDecoder, div_by_world_size
from fairseq import utils
import torch.nn as nn
from smoe.moe_layer import MoELayer
from smoe.htop2gate import HTop2Gate

class SmoeDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(args, dictionary, embed_tokens)
        moe_freq = max(getattr(args, 'decoder_moe_freq', 0), getattr(args, 'moe_freq', 0))
        assert moe_freq==2, 'currently only supports moe-freq=2'
        self.decoder_layers=[self.build_decoder_layer(args, is_moe_layer=False) 
                        for _ in range(args.decoder_layers//2)]
        self.self_cross_attn_layers=[
            [self.build_self_attn_layer(args), self.build_cross_attn_layer(args)] for _ in range(args.decoder_layers//2)
        ]

        self.ffn_layers=[self.build_ffn_layer(args),]
        self.moe_layers=[]
        for i in range(args.decoder_layers//2):
            layer_idx=i if i!=(args.decoder_layers//2-1) else -1
            self.moe_layers.append(self.build_moe_layer(args, layer_idx))
        del self.layers

        self.layers=[]
        for i,l in enumerate(self.decoder_layers):
            self.layers.append(self.decoder_layers[i])
            self.layers.append(self.self_cross_attn_layers[i][0]),
            self.layers.append(self.self_cross_attn_layers[i][1])
            self.layers.append(self.moe_layers[i])

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        tgt_lang_id: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
            self_attn_padding_mask (torch.Tensor, optional): precomputed padding
                mask for self-attention (default None will recompute mask)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            tgt_lang_id=tgt_lang_id,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )

        if not self.training and getattr(self.args, 'record_token_expert', False):
            self.collect_expert_choices(prev_output_tokens)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        tgt_lang_id: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            tgt_lang_id=tgt_lang_id,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        tgt_lang_id: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        token_embeddings: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        A scriptable subclass of this class has an extract_features method and calls
        super().extract_features, but super() is not supported in torchscript. A copy
        of this function is made to be used in the subclass instead.
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # compute self-attention padding mask (involves device-to-host transfer,
        # so put it at the top of the forward)
        if (
            self_attn_padding_mask is None
            and (
                self.cross_self_attention
                or prev_output_tokens.device.type == "xla"
                or prev_output_tokens.eq(self.padding_idx).any()
            )
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed tokens and positions
        x, _ = self.forward_embedding(prev_output_tokens, token_embeddings, incremental_state)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []
            # l_aux = [encoder_out["l_aux"]] if "l_aux" in encoder_out else []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if isinstance(layer, DecoderMoeLayer):
                x, layer_attn, _, l_aux_i = layer(
                    x,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    tgt_lang_id,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            else:
                x, layer_attn, _, l_aux_i = layer(
                    x,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            l_aux.append(l_aux_i)
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "l_aux": l_aux}

    def build_cross_attn_layer(self, args):
        layer = DecoderCrossAttnLayer(args)
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

    def build_ffn_layer(self, args):
        layer = DecoderFFNLayer(args)
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

    def build_self_attn_layer(self, args):
        layer = DecoderSelfAttnLayer(args)
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
    
    def build_moe_layer(self, args, layer_idx):
        layer = DecoderMoeLayer(args, layer_idx)
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

class DecoderSelfAttnLayer(TransformerDecoderLayer):
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
        self.normalize_before = args.decoder_normalize_before

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x, attn, None, None

class DecoderCrossAttnLayer(TransformerDecoderLayer):
    def __init__(self, args) -> None:
        super().__init__(args)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)
        
        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        else:
            attn = None
        return x, attn, None, None

class DecoderFFNLayer(nn.Module):
    # TODO: simplify the module
    def __init__(self, args, ) -> None:
        super().__init__()
        self.is_moe_layer=False
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        ffn_dim = args.decoder_ffn_embed_dim
        self.layer_norm=LayerNorm(self.embed_dim)
        self.ffn=FeedForwardNetwork(args, self.embed_dim, ffn_dim, self.dropout_module)
    
    def forward(
        self,
        x,
        *args, **kwargs
    ):
        res=x
        x=self.layer_norm(x)
        x=self.ffn(x)
        x=res+x
        return x, None, None, None
    
    def upgrade_state_dict_named(self, *args, **kwargs):
        pass

class DecoderMoeLayer(TransformerDecoderLayer):
    # TODO: simplify the module
    def __init__(self, args, layer_idx):        
        self.layer_idx=layer_idx
        super().__init__(args, is_moe_layer=True)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if 'attn' in n and hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        tgt_lang_id: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        layer_idx=self.moe_layer.layer_idx
        norm_before=self.normalize_before

        res=x
        if norm_before:
            x = self.final_layer_norm(x)
        
        x = x.transpose(0, 1) # batch_size, seq_len, model_dim
        if getattr(self.args, "use_moe_pad_mask", False):
            x, l_aux = self.moe_layer(x, input_padding_mask=self_attn_padding_mask, tgt_lang_id=tgt_lang_id)
        else:
            x, l_aux = self.moe_layer(x, tgt_lang_id=tgt_lang_id)
        x = x.transpose(0, 1) # seq_len, batch_size, model_dim
        x = self.residual_connection(x, res)
        
        if not norm_before:
            x = self.final_layer_norm(x)
        return x, None, None, l_aux
    
    def build_moe_layer(self, gate, experts, args):
        return MoELayer(gate, experts, args, self.embed_dim, self.layer_idx, autoregressive=True)
    
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
