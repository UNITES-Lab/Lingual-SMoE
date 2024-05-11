import math
import time
from smoe.utils import is_inf, is_nan
import torch.distributed as dist
import torch
import fairseq.utils as utils
from torch import Tensor, nn
from fairseq.models.transformer import div_by_world_size
import functools
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from typing import Any, List, Union
from fairseq.modules.moe.moe_layer import MOELayer
import fairseq.distributed.utils as  distributed_utils
from fairseq.modules.moe.moe_layer import _AllToAll
from torch.nn.functional import pad
import logging
logger = logging.getLogger(__name__)

def fsdp_wrap_expert(args, process_group, experts):
    # Wrap MoE layer with FSDP using a process group with all replicated ranks
    world_size = distributed_utils.get_data_parallel_group().size()
    pg_size = process_group.size()
    num_experts = world_size/pg_size
    
    for i, expert in enumerate(experts):
        experts[i] = fsdp_wrap(
            expert, process_group=process_group, min_num_params=0
        )
    if getattr(args, "moe_normalize_expert_grad", "world_size") == "sqrt_world_size":
        expert_normalization_term = math.sqrt(num_experts)
    else:
        expert_normalization_term = num_experts
    for p in experts.parameters():
        p.expert = True
        p.register_hook(functools.partial(div_by_world_size, expert_normalization_term))

    return experts

class MoELayer(MOELayer):
    def __init__(self, gate, experts, args, d_model, layer_idx, autoregressive=False) -> None:
        global_experts=experts['global']
        
        all_experts=nn.ModuleList([])
        all_experts.extend(global_experts)
        
        setattr(gate, 'autoregressive', autoregressive)
        self.autoregressive=autoregressive
        
        super().__init__(gate, all_experts, args)
        self.num_local_experts=args.moe_expert_count//self.all2all_size
        
        for p in global_experts.parameters():
            assert p.expert
        
        self.all2all_group = get_all2all_group()
        self.node_group = get_node_group()
        
        self.all2all_size=distributed_utils.get_world_size(self.all2all_group)
        self.local_world_size=distributed_utils.get_world_size(self.node_group)
        assert self.local_world_size<=8
        assert self.all2all_size%self.local_world_size==0
        self.node_num=self.all2all_size//self.local_world_size
        rank=distributed_utils.get_global_rank()
        self.node_rank, self.local_rank=rank//self.local_world_size, rank%self.local_world_size
        
        self.use_hmoe_gate=args.hmoe_gate
        self.add_lang_loss=args.add_lang_loss
        self.num_tasks=args.num_tasks

        self.experts=fsdp_wrap_expert(args, self.expert_group, self.experts)
        if getattr(args, 'layernorm_after_moe_layer', False):
            self.post_layernorm=nn.LayerNorm(d_model, elementwise_affine=False)
        else:
            self.post_layernorm=None
        self.layer_idx=layer_idx
        self.has_print_examples=False

        
    def forward(self, *input: Tensor, res_connection=None, input_padding_mask=None, tgt_lang_id=None, **kwargs: Any):
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        if input_padding_mask is None:
            input_padding_mask=torch.full([input.shape[0], input.shape[1]], False, device=input.device)
     
        return self.forward_emb(input, input_padding_mask, tgt_lang_id, *kwargs)
          
    def forward_emb(self, input, input_padding_mask, tgt_lang_id, *kwargs):   
        if input_padding_mask.shape[1] != input.shape[1]:
            input_padding_mask=input_padding_mask[:,-1]
        reshaped_input=input.reshape(-1, input.shape[-1])
        input_shape = list(reshaped_input.shape)
        reshaped_input_padding_mask=input_padding_mask.reshape(-1)
        
        expected_dim = int(distributed_utils.all_reduce(
            input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
            group=dist.group.WORLD,
            op="max",
        ).item())

        padded_input = torch.zeros(
            (expected_dim, input_shape[1]),
            dtype=input.dtype, layout=input.layout, device=input.device)
        padded_input[:input_shape[0], :] = reshaped_input
        reshaped_input = padded_input

        # pad tgt_lang_id to (max batch*seq_len in all experts, model_dim)
        if tgt_lang_id is not None and self.use_hmoe_gate:
            if self.add_lang_loss and self.training:
                global_lang_emb = tgt_lang_id[-self.num_tasks:,:]
                tgt_lang_id = tgt_lang_id[:-self.num_tasks,:]
            else:
                global_lang_emb=None
            token_tgt_id = tgt_lang_id.repeat(1,input.shape[1],1) # (batch, seq_len, task emb dim)
            reshaped_tgt_id = token_tgt_id.reshape(-1, tgt_lang_id.shape[-1]) # (batch*seq_len, task emb dim)
            padded_tgt_id = torch.zeros(
                (expected_dim, reshaped_tgt_id.shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_tgt_id[:input_shape[0], :] = reshaped_tgt_id
            reshaped_tgt_id = padded_tgt_id
            reshaped_tgt_id = reshaped_tgt_id.float()
        else:
            global_lang_emb = None
            reshaped_tgt_id = None
 
        padded_input_padding_mask = torch.ones(
            (expected_dim,), dtype=torch.bool, device=padded_input.device
        )
        if reshaped_input_padding_mask is not None:
            padded_input_padding_mask[:input_shape[0]] = reshaped_input_padding_mask
        else:
            padded_input_padding_mask[:input_shape[0]] = False
        reshaped_input_padding_mask = padded_input_padding_mask
        
    
        combined_output, l_aux= self.moe(reshaped_input, reshaped_input_padding_mask, 
                                         reshaped_tgt_id,
                                         global_lang_emb, 
                                         self.all2all_group)
        
        result = combined_output[:input_shape[0], :]
        result = result.reshape_as(input)
        self.record_all_to_all_stats()
        return result, l_aux

    def moe(self, tokens, input_mask, reshaped_tgt_id, global_lang_emb, group, route_scores=None, experts=[]):
        _, d_model=tokens.shape
        if group is not None:
            world_size=distributed_utils.get_world_size(group)
        else:
            world_size=1
        
        if self.use_hmoe_gate:
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
                lang_input=reshaped_tgt_id, 
                token_input=tokens, 
                global_lang_emb=global_lang_emb, 
                logits=route_scores, mask=input_mask)
        else:
            # router is fairseq/modules/moe/top2gate.py
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
                input=tokens, logits=route_scores, mask=input_mask)

        dispatch_mask = dispatch_mask.to(tokens.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        assert tokens.size() == (S, d_model)
        # einsum("sec,sm->ecm")
        dispatched_input = torch.mm(dispatch_mask.view(E*C, S), tokens)  # -> (E*C),M

        if world_size!=1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input, group)

        dispatched_input = dispatched_input.reshape(world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []

        if len(experts)==0:
            experts=self.experts
        for chunk, expert in zip(chunks, experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        if self.post_layernorm is not None:
            expert_output=self.post_layernorm(expert_output)
        if world_size!=1:
            expert_output = self.all_to_all_wrapper(expert_output, group)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(world_size * self.num_local_experts, -1, d_model)

        # einsum("sec,ecm->sm")
        combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, d_model))
   
        assert tokens.shape==combined_output.shape
        assert not is_inf(combined_output) and not is_nan(combined_output)
        return combined_output, l_aux

    def all_to_all_wrapper(self, input: Tensor, group=None, input_splits=None, output_splits=None):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        if group is None:
            group=self.all2all_group

        output = _AllToAll.apply(group, input, input_splits, output_splits)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output
    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

def get_all2all_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_all2all_group, "_all2all_groups"):
            world_size=distributed_utils.get_global_world_size()
            all2all_groups=dist.new_group(list(range(world_size)))
            get_all2all_group._all2all_groups=all2all_groups
        return get_all2all_group._all2all_groups

def get_node_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_node_group, "_node_groups"):
            world_size=distributed_utils.get_global_world_size()
            local_world_size=torch.cuda.device_count()
            assert local_world_size>1
            node_num=world_size//local_world_size
            global_rank=distributed_utils.get_global_rank()
            node_rank=global_rank//local_world_size
            for i in range(node_num):
                ranks=list(range(i*local_world_size, (i+1)*local_world_size))
                node_groups=dist.new_group(ranks)
                if i==node_rank:
                    get_node_group._node_groups=node_groups
        return get_node_group._node_groups

def share_mem(action='write', key:str=None, value:object=None, ):
    if not hasattr(share_mem, '_memory'):
        share_mem._memory=dict()    
    if action=='read':
        assert key in share_mem._memory
        return share_mem._memory[key]
    else:
        if isinstance(value, torch.Tensor):
            share_mem._memory[key]=value.detach()
        else:
            share_mem._memory[key]=value

